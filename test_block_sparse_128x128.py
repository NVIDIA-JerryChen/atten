"""
demo_fa4.py - FA4 Block Sparse Attention Demo for B200 (Blackwell/SM100)

对比测试:
- FA4 Block Sparse Attention (flash_attn.cute.interface)
- PyTorch SDPA (dense)
- FlexAttention (sparse)

测试 Shape:
- head_dim=64: [8, 130560, 4, 64], [64, 65280, 2, 64]
- head_dim=128: [2, 1590, 4, 128], [2, 8160, 4, 128]

Usage:
    python demo_fa4.py --device cuda:0 --sparsity 0.5
    python demo_fa4.py --accuracy-only
    python demo_fa4.py --performance-only --log
"""

import sys
import os

# 把系统包路径放最前面
# sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')

import argparse
from termcolor import colored
import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Callable
from datetime import datetime
import sys
import nvtx
import pandas as pd
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)

# FA4 cute interface
# try:
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch 
from flash_attn.cute import utils as cute_utils
import cutlass
import cutlass.cute as cute
FA4_AVAILABLE = True
# except ImportError as e:
#     FA4_AVAILABLE = False
#     FA4_IMPORT_ERROR = str(e)
#     print(f"Warning: FA4 cute interface not available: {e}")
#     # 定义占位类型，避免类型注解报错
#     from typing import NamedTuple
#     class BlockSparseTensorsTorch(NamedTuple):
#         mask_block_cnt: torch.Tensor
#         mask_block_idx: torch.Tensor
#         full_block_cnt: Optional[torch.Tensor] = None
#         full_block_idx: Optional[torch.Tensor] = None
    
#     def fast_sampling(fn):
#         return fn

# 预编译 flex_attention
torch._dynamo.reset()
flex_attention_compiled = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)

WARMUP_ITERATIONS = 10
PERF_BENCHMARK_ITERATIONS = 20
BLOCK_USER = 128


class TeeLogger:
    """将输出同时打印到控制台和日志文件"""
    
    def __init__(self, log_file_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.closed = False
    
    def write(self, message: str) -> None:
        self.terminal.write(message)
        if not self.closed:
            try:
                self.log_file.write(message)
                self.log_file.flush()
            except (OSError, ValueError):
                self.closed = True
    
    def flush(self) -> None:
        self.terminal.flush()
        if not self.closed:
            try:
                self.log_file.flush()
            except (OSError, ValueError):
                self.closed = True
    
    def close(self) -> None:
        self.closed = True
        if self.log_file:
            self.log_file.close()


def generate_block_mask_128(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    sparsity: float,
    device: torch.device | str,
) -> torch.Tensor:
    """
    生成 128x128 block mask，满足以下条件：
    1. 总体 sparsity 精确等于目标值（强约束）
       - sparsity = 被跳过的比例，density = 1 - sparsity = 被计算的比例
    2. 相邻行对 (2k, 2k+1) 共享相同的 mask pattern
    3. 每 pair 至少有 1 个 True
    4. 不同 pair 的 k 值可以不同
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    num_blocks = math.ceil(seq_len / BLOCK_USER)
    # num_q_pairs = (num_blocks + 1) // 2  # Q 方向的行对数量
    num_q_pairs = num_blocks

    # density = 1 - sparsity（被计算的比例）
    # sparsity 80% → density 20% → 只计算 20% 的 blocks
    density = 1.0 - sparsity

    # 目标总 True 数量（强约束）
    total_elements = num_q_pairs * num_blocks
    total_true_target = int(density * total_elements)
    total_true_target = max(total_true_target, num_q_pairs)  # 每 pair 至少 1 个
    total_true_target = min(total_true_target, total_elements)  # 不超上限

    # 生成随机分数
    scores = torch.rand((batch_size, num_heads, num_q_pairs, num_blocks), device=device)

    # Step 1: 先为每 pair 选择分数最高的 1 个位置（保证每 pair 至少有 1 个）
    _, first_indices = scores.max(dim=-1, keepdim=True)  # (B, H, num_q_pairs, 1)
    pair_mask = torch.zeros_like(scores, dtype=torch.bool)
    pair_mask.scatter_(3, first_indices, True)

    # Step 2: 剩余需要选择的数量
    remaining = total_true_target - num_q_pairs

    if remaining > 0:
        # 将已选中的位置分数设为 -inf，避免重复选择
        scores_masked = scores.clone()
        scores_masked.scatter_(3, first_indices, float("-inf"))

        # 展平并选择 top-remaining
        scores_flat = scores_masked.view(batch_size, num_heads, -1)
        _, top_indices = torch.topk(scores_flat, remaining, dim=-1)

        # 更新 mask
        pair_mask_flat = pair_mask.view(batch_size, num_heads, -1)
        pair_mask_flat.scatter_(2, top_indices, True)
        pair_mask = pair_mask_flat.view(batch_size, num_heads, num_q_pairs, num_blocks)

    # 扩展到 (batch_size, num_heads, num_blocks, num_blocks)
    # 每对行重复 2 次（第 2k 行和第 2k+1 行相同）
    # block_mask = pair_mask.repeat_interleave(2, dim=2)

    # 如果 num_blocks 是奇数，截断多余的行
    block_mask = pair_mask[:, :, :num_blocks, :]

    return block_mask


# def merge_block_mask_128_to_256_random(block_mask_128: torch.Tensor) -> torch.Tensor:
#     """将 128x128 block mask 随机合并成 256x128 粗粒度 mask。"""
#     if block_mask_128.ndim != 4:
#         raise ValueError(f"block_mask_128 must be 4D, got shape {block_mask_128.shape}")

#     batch_size, num_heads, num_q_blocks, num_k_blocks = block_mask_128.shape
#     if num_q_blocks % 2 == 1:
#         pad_row = block_mask_128[:, :, -1:, :]
#         block_mask_128 = torch.cat([block_mask_128, pad_row], dim=2)
#         num_q_blocks += 1

#     paired = block_mask_128.view(batch_size, num_heads, num_q_blocks // 2, 2, num_k_blocks)
#     selector = torch.rand(
#         (batch_size, num_heads, num_q_blocks // 2, 1),
#         device=block_mask_128.device,
#     ) < 0.5
#     merged = torch.where(selector, paired[:, :, :, 0, :], paired[:, :, :, 1, :])
#     return merged.to(torch.uint8)


def generate_upsampled_mask_mod(
    binary_mask: torch.Tensor, block_size: int = 128
) -> Callable:
    """
    从 binary block mask 生成 mask_mod 函数用于 create_block_mask
    
    Args:
        binary_mask: (B, H, num_q_blocks, num_k_blocks) 的 0-1 mask
        block_size: block 大小
    
    Returns:
        mask_mod: 用于 create_block_mask 的函数
    """
    def upsampled_mask_mod(b, h, q_idx, kv_idx):
        downsampled_q_idx = q_idx // block_size
        downsampled_kv_idx = kv_idx // block_size
        return binary_mask[b, h, downsampled_q_idx, downsampled_kv_idx]
    return upsampled_mask_mod


def create_block_mask_from_binary(
    binary_mask: torch.Tensor,
    seq_len: int,
    q_block_size: int,
    k_block_size: int,
    mask_mod: Callable,
) -> BlockMask:
    """
    从 binary block mask 直接构造 FlexAttention 的 BlockMask，绕过 create_block_mask 的 O(seq²) 内存问题。
    
    Args:
        binary_mask: (B, H, num_q_blocks, num_k_blocks) 的 0-1 mask
        seq_len: 序列长度
        q_block_size: Q 方向 block 大小
        k_block_size: K 方向 block 大小
        mask_mod: FlexAttention 需要的 mask_mod 函数
    
    Returns:
        BlockMask: FlexAttention 的 BlockMask 对象
    """
    B, H, num_q_blocks, num_k_blocks = binary_mask.shape
    device = binary_mask.device
    
    # kv_num_blocks: 每个 Q block 需要计算多少个 KV blocks
    kv_num_blocks = binary_mask.sum(dim=-1).to(torch.int32)  # (B, H, num_q_blocks)
    
    # kv_indices: 需要计算的 KV block 索引
    max_kv_per_q = kv_num_blocks.max().item()
    if max_kv_per_q == 0:
        max_kv_per_q = 1  # 至少分配 1 个位置
    
    kv_indices = torch.zeros(B, H, num_q_blocks, max_kv_per_q, dtype=torch.int32, device=device)
    
    # 填充 kv_indices（向量化实现）
    for b in range(B):
        for h in range(H):
            for q in range(num_q_blocks):
                mask_row = binary_mask[b, h, q]  # (num_k_blocks,)
                indices = torch.nonzero(mask_row, as_tuple=False).squeeze(-1)  # 非零位置
                if indices.numel() > 0:
                    kv_indices[b, h, q, :indices.numel()] = indices.to(torch.int32)
    
    # 使用 FlexAttention 使用的 BLOCK_SIZE（128x128）
    flex_block_size = 128
    
    return BlockMask(
        seq_lengths=(seq_len, seq_len),
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        q_num_blocks=None,
        q_indices=None,
        full_q_num_blocks=None,
        full_q_indices=None,
        BLOCK_SIZE=(flex_block_size, flex_block_size),
        mask_mod=mask_mod,
    )


def create_cute_block_sparse_mask_mod(block_size: int = 128):
    """
    创建 CuTe JIT 格式的 block sparse mask_mod 函数(用于 fine mode)
    
    这个函数从 aux_tensors[0] 读取 128x128 粒度的 block mask，
    并在 element level 做精确的 masking。
    
    Args:
        block_size: block 大小(默认 128)
    
    Returns:
        mask_mod: CuTe JIT 格式的 mask_mod 函数
    """
    if not FA4_AVAILABLE:
        raise RuntimeError("FA4 not available, cannot create CuTe mask_mod")
    
    # @fast_sampling
    @cute.jit
    def cute_block_sparse_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors: list,
    ) -> cute.TensorSSA:
        """
        CuTe JIT mask_mod 函数，从 aux_tensors[0] 读取 block mask
        
        aux_tensors[0]: (B, H, num_q_blocks, num_k_blocks) 的 0-1 mask
        """
        block_mask = aux_tensors[0]
        block_size_ssa = cute_utils.scalar_to_ssa(block_size, cutlass.Int32)
        
        # 计算 block 索引
        q_block_idx = m_idx // block_size_ssa
        k_block_idx = n_idx // block_size_ssa
        
        # 从 block mask 中读取值
        mask_value = cute_utils.scalar_to_ssa(
            block_mask[batch[0], head[0], q_block_idx[0], k_block_idx[0]], 
            cutlass.Int32
        )
        
        # 返回 True 如果 mask 值为 1
        return mask_value > cute_utils.scalar_to_ssa(0, cutlass.Int32)
    
    return cute_block_sparse_mask


def convert_block_mask_to_fa4_format(
    block_mask_flex: BlockMask,
) -> BlockSparseTensorsTorch:
    """
    将 FlexAttention 的 BlockMask 转换为 FA4 BlockSparseTensorsTorch 格式
    
    Args:
        block_mask_flex: FlexAttention 的 BlockMask 对象
    
    Returns:
        BlockSparseTensorsTorch: FA4 所需的 block sparse tensors
    """
    (
        _seq_q,
        _seq_k,
        kv_mask_cnt,
        kv_mask_idx,
        full_kv_cnt,
        full_kv_idx,
        q_mask_cnt,
        q_mask_idx,
        full_q_cnt,
        full_q_idx,
        *_,
    ) = block_mask_flex.as_tuple()
    
    return BlockSparseTensorsTorch(
        mask_block_cnt=kv_mask_cnt,
        mask_block_idx=kv_mask_idx,
        full_block_cnt=full_kv_cnt,
        full_block_idx=full_kv_idx,
    )


def convert_binary_mask_to_fa4_format(
    block_mask_binary: torch.Tensor,
    q_stage: int = 1,
) -> BlockSparseTensorsTorch:
    """
    直接从 binary block mask 创建 FA4 BlockSparseTensorsTorch 格式
    不需要通过 FlexAttention，避免大 seq_len 时的 OOM
    
    对于 non-causal 128x128 block sparse (q_stage=1):
    - 所有选中的 block 都是 FULL block（不需要 element-level mask）
    - full_block_cnt/idx 包含选中的 block
    - mask_block_cnt/idx 应为空
    
    Args:
        block_mask_binary: (B, H, num_q_blocks, num_k_blocks) 的 0-1 mask
        q_stage: FA4 的 q_stage 参数，SM100 上 q_stage=1 表示128x128 粒度，目前已支持 128x128 粒度
    
    Returns:
        BlockSparseTensorsTorch: FA4 所需的 block sparse tensors
    """
    B, H, num_q_blocks, num_k_blocks = block_mask_binary.shape
    device = block_mask_binary.device
    
    # FA4 期望的 q blocks 数量是 ceil(num_q_blocks / q_stage)
    # 需要合并相邻的 q blocks
    if q_stage > 1:
        # 将相邻的 q_stage 个 q blocks 合并(取 OR)
        # 先 padding 使得 num_q_blocks 是 q_stage 的倍数
        pad_q = (q_stage - num_q_blocks % q_stage) % q_stage
        if pad_q > 0:
            padding = torch.zeros(B, H, pad_q, num_k_blocks, dtype=block_mask_binary.dtype, device=device)
            block_mask_binary = torch.cat([block_mask_binary, padding], dim=2)
        
        # reshape 并合并
        new_num_q_blocks = block_mask_binary.shape[2] // q_stage
        block_mask_binary = block_mask_binary.view(B, H, new_num_q_blocks, q_stage, num_k_blocks)
        # 合并方式：如果任何一个子 block 有 mask，合并后的 block 就有 mask
        block_mask_binary = block_mask_binary.any(dim=3).to(torch.uint8)
    
    B, H, num_q_blocks_effective, num_k_blocks = block_mask_binary.shape
    
    # 计算每个 query block 对应的有效 key block 数量
    # 对于 non-causal，所有选中的 block 都是 FULL block
    # shape: (B, H, num_q_blocks_effective)
    full_block_cnt = block_mask_binary.sum(dim=-1).to(torch.int32)
    
    # 创建 full_block_idx: (B, H, num_q_blocks_effective, num_k_blocks)
    # 使用向量化操作获取每个 query block 的有效 key block 索引
    # 将有效的 k block 索引排在前面，无效的位置填 0
    positions = torch.arange(num_k_blocks, device=device).view(1, 1, 1, -1).expand(B, H, num_q_blocks_effective, -1)
    masked_positions = torch.where(
        block_mask_binary.bool(),
        positions.float(),
        torch.tensor(float('inf'), device=device)
    )
    sorted_indices = masked_positions.sort(dim=-1).values
    sorted_indices = torch.where(
        sorted_indices == float('inf'),
        torch.zeros_like(sorted_indices),
        sorted_indices
    )
    full_block_idx = sorted_indices.to(torch.int32)
    
    # 对于 non-causal 128x128 block sparse，mask_block 应为空
    # （除了对于超出 seqlen 的 block，其余block 内部不需要 element-level masking）
    mask_block_cnt = torch.zeros_like(full_block_cnt)
    mask_block_idx = torch.zeros_like(full_block_idx)
    
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
    )


def expand_block_mask(
    block_mask: torch.Tensor, 
    seq_len: int, 
    q_block_size: int = 128,
    k_block_size: int = 128,
) -> torch.Tensor:
    """
    将 block_mask 扩展为完整的 attention mask
    
    Args:
        block_mask: (B, H, num_q_blocks, num_k_blocks) 的 0-1 mask
        seq_len: 序列长度
        q_block_size: Q 方向的 block 大小
        k_block_size: K 方向的 block 大小
    
    Returns:
        expanded_mask: (B, H, seq_len, seq_len) 的 attention mask
    """
    B, H, nb_q, nb_k = block_mask.shape
    expanded = block_mask.repeat_interleave(q_block_size, dim=2).repeat_interleave(
        k_block_size, dim=3
    )
    return expanded[:, :, :seq_len, :seq_len]


@nvtx.annotate("pytorch_reference.forward", color="green")
def pytorch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """PyTorch 参考实现"""
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, 0.0)
    out = torch.matmul(attn, v)
    return out


@nvtx.annotate("pytorch_reference_chunked.forward", color="green")
def pytorch_reference_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: Optional[torch.Tensor] = None,
    q_block_size: int = 128,
    k_block_size: int = 128,
    sm_scale: Optional[float] = None,
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    PyTorch 参考实现 - 分块版本，节省内存
    
    关键改进：直接接受 block_mask，动态生成每个 chunk 的 mask slice，
    避免创建完整的 (B, H, seq, seq) expanded mask。
    
    Args:
        q, k, v: (B, H, N, D) 格式的输入
        block_mask: (B, H, num_q_blocks, num_k_blocks) 的 block-level mask
        q_block_size: Q 方向的 block 大小
        k_block_size: K 方向的 block 大小
        sm_scale: softmax scale
        chunk_size: 每次处理的 Q 序列长度
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    B, H, N, D = q.shape
    output = torch.zeros_like(q)
    
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        q_chunk = q[:, :, i:end_i, :]
        scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * sm_scale  # (B, H, chunk, N)
        
        if block_mask is not None:
            # 动态生成当前 chunk 的 mask，不存储完整 mask
            chunk_len = end_i - i
            # 计算当前 chunk 涉及的 Q block 范围
            q_block_start = i // q_block_size
            q_block_end = (end_i - 1) // q_block_size + 1
            
            # 只扩展当前 chunk 需要的 mask 部分
            # block_mask: (B, H, num_q_blocks, num_k_blocks)
            chunk_block_mask = block_mask[:, :, q_block_start:q_block_end, :]  # (B, H, chunk_blocks, num_k_blocks)
            
            # 扩展到 element level
            expanded_q = chunk_block_mask.repeat_interleave(q_block_size, dim=2)  # (B, H, chunk_blocks*q_block_size, num_k_blocks)
            expanded_kv = expanded_q.repeat_interleave(k_block_size, dim=3)  # (B, H, chunk_blocks*q_block_size, N_padded)
            
            # 裁剪到实际需要的大小
            local_start = i - q_block_start * q_block_size
            mask_chunk = expanded_kv[:, :, local_start:local_start + chunk_len, :N]
            
            scores = scores.masked_fill(mask_chunk == 0, float("-inf"))
            del chunk_block_mask, expanded_q, expanded_kv, mask_chunk
        
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, 0.0)
        output[:, :, i:end_i, :] = torch.matmul(attn, v)
        del scores, attn
    
    return output


@nvtx.annotate("fa4_sparse_attention.forward", color="blue")
def fa4_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_sparse_tensors: BlockSparseTensorsTorch,
    m_block_size: int = 128,
    n_block_size: int = 128,
) -> torch.Tensor:
    """
    使用 FA4 cute interface 计算 block sparse attention
    
    Args:
        query: (B, H, S, D) 格式
        key: (B, H, S, D) 格式
        value: (B, H, S, D) 格式
        block_sparse_tensors: FA4 block sparse tensors
        m_block_size: Q 方向的 tile 大小 (对于 SM100, 有效粒度 = q_stage * m_block_size)
        n_block_size: K 方向的 tile 大小
    
    Returns:
        output: (B, H, S, D) 格式
    """
    if not FA4_AVAILABLE:
        raise RuntimeError(f"FA4 cute interface not available: {FA4_IMPORT_ERROR}")
    
    # FA4 期望输入格式为 (B, S, H, D)，需要从 (B, H, S, D) 转换
    B, H, S, D = query.shape
    q = query.transpose(1, 2)  # (B, S, H, D) only change layout
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    
    with nvtx.annotate("fa4_fwd_kernel", color="yellow"):
        out, lse = _flash_attn_fwd(
            q=q,
            k=k,
            v=v,
            softmax_scale=None,  # 自动计算
            causal=False,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            mask_mod=None,  # 不使用 mask_mod，完全依赖 block_sparse_tensors
            block_sparse_tensors=block_sparse_tensors,
            aux_tensors=None,
            return_lse=False,
        )
    
    return out.transpose(1, 2)


@nvtx.annotate("flex_sparse_attention.forward", color="green")
def flex_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: BlockMask,
    flex_attention_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """使用 FlexAttention 计算 sparse attention"""
    with torch.cuda.device(query.device):
        if flex_attention_fn is not None:
            hidden_states = flex_attention_fn(query, key, value, block_mask=block_mask)
        else:
            hidden_states = flex_attention(query, key, value, block_mask=block_mask)
    return hidden_states


@nvtx.annotate("pytorch_sdpa.forward", color="red")
def pytorch_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """PyTorch SDPA (dense attention)"""
    return F.scaled_dot_product_attention(query, key, value)


def validate_accuracy(
    ref_data: torch.Tensor,
    test_data: torch.Tensor,
    message: str = "",
    rtol: float = 1e-4,
    atol: float = 1e-2,
    verbose: bool = False,
) -> bool:
    """误差验证函数"""
    mae_tol = rmse_tol = 1e-3
    ref_data_f32 = ref_data.to(torch.float32)
    test_data_f32 = test_data.to(torch.float32)
    abs_err = (ref_data_f32 - test_data_f32).abs()
    rel_err = abs_err / (ref_data_f32.abs() + 1e-12)
    
    metrics = {}
    metrics['max_abs_err'] = abs_err.max().item()
    metrics['mae'] = abs_err.mean().item()
    metrics['rmse'] = torch.sqrt((abs_err ** 2).mean()).item()
    metrics['max_rel_err'] = rel_err.max().item()
    metrics['mean_rel_err'] = rel_err.mean().item()
    
    ref_flat = ref_data_f32.flatten()
    test_flat = test_data_f32.flatten()
    metrics['cosine_sim'] = F.cosine_similarity(
        ref_flat.unsqueeze(0), test_flat.unsqueeze(0)
    ).item()
    
    signal_power = (ref_data_f32 ** 2).mean()
    noise_power = (abs_err ** 2).mean()
    metrics['snr_db'] = 10 * torch.log10(signal_power / (noise_power + 1e-12)).item()
    
    metrics['allclose'] = torch.allclose(ref_data_f32, test_data_f32, rtol=rtol, atol=atol)
    within_tol = (abs_err <= atol + rtol * ref_data_f32.abs()).sum().item()
    metrics['pass_rate'] = within_tol / ref_data_f32.numel() * 100
    
    validation_results = (
        (metrics['allclose'] and metrics['pass_rate'] >= 95) and
        metrics['max_abs_err'] < atol and
        metrics['mae'] < mae_tol and
        metrics['rmse'] < rmse_tol
    )
    
    print("🔍 " + ">" * 5 + f" {message}")
    if validation_results:
        print(colored("✅ Accuracy Validation Passed", "green"))
    else:
        print(colored("❌ Accuracy Validation Failed", "red"))
    print(f"  Absolute Errors (atol={atol}, mae_tol={mae_tol}, rmse_tol={rmse_tol}):")
    print(f"    Max:    {metrics['max_abs_err']:.6e}")
    print(f"    Mean:   {metrics['mae']:.6e}")
    print(f"    RMSE:   {metrics['rmse']:.6e}")
    if verbose:
        print(f"  Cosine Similarity: {metrics['cosine_sim']:.8f}")
        print(f"  SNR: {metrics['snr_db']:.2f} dB")
        print(f"  Pass Rate: {metrics['pass_rate']:.2f}%")
    print("")
    return validation_results


def run_single_test(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int = 128,
    sparsity: float = 0.5,
    device: str = "cuda",
    run_accuracy: bool = True,
    run_performance: bool = True,
    # mode: str = "coarse",
    test_case_idx: int = 0,
) -> Optional[dict[str, str]]:
    """
    运行单个测试 case
    
    Args:
        mode: "coarse" - 256x128 粗粒度(无 mask_mod)
              "fine" - 128x128 细粒度(使用 CuTe JIT mask_mod)
    """
    print("=" * 70)
    print(colored(
        f"Test Case {test_case_idx}: B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}",
        "cyan"
    ))
    # SM100 FA4 硬编码限制：
    # - m_block_size 必须是 128（TMEM 32x32 原子操作要求）
    # - q_stage=2 是硬编码的（当 seqlen > m_block_size）
    # - 有效 Q 粒度 = q_stage * m_block_size = 2 * 128 = 256
    # - 因此 SM100 上 FA4 block sparse 只支持 256x128 粒度
    # 
    # fine mode 在 SM100 上不可用（m_block_size=64 不支持）
    # 两种 mode 使用相同的 256x128 配置，fine mode 仅作为标记保留
    # if mode == "fine":
    #     print(colored(
    #         "WARNING: Fine mode (128x128) not supported on SM100. "
    #         "Using coarse mode (256x128) instead.", "yellow"
    #     ))
    #     mode = "coarse"
    
    mode = 'fine'
    mode_desc = "fine (128x128)"
    print(f"Mode: {mode_desc}, Block Size: {block_size}, Sparsity: {sparsity * 100:.1f}%")
    print("=" * 70)
    
    # SM100 唯一支持的配置: 256x128
    # m_block_size=128, q_stage=2 -> 有效 Q 粒度 = 256
    q_block_size = 128
    k_block_size = 128
    m_block_size = 128
    n_block_size = 128

    # 生成 128x128 block mask，再随机合并为 256x128
    block_mask_128 = generate_block_mask_128(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        sparsity=sparsity,
        device=device,
    )
    # block_mask_binary = merge_block_mask_128_to_256_random(block_mask_128)
    block_mask_binary = block_mask_128
    
    mask_density = block_mask_binary.sum().item() / block_mask_binary.numel()
    print(f"Actual mask density: {mask_density * 100:.2f}%")
    print(f"Block mask shape: {block_mask_binary.shape}")
    
    # 生成 QKV 张量 (B, H, S, D) 格式
    qkv_shape = (batch_size, num_heads, seq_len, head_dim)
    query = torch.randn(qkv_shape, dtype=torch.bfloat16, device=device)
    key = torch.randn(qkv_shape, dtype=torch.bfloat16, device=device)
    value = torch.randn(qkv_shape, dtype=torch.bfloat16, device=device)
    
    # 创建 FlexAttention block mask - 使用 _compile=True 避免 materialize full mask
    block_mask_flex = None
    flex_mask_created = False
    
    # 创建 FlexAttention mask_mod（用于 flex_attention 运行时）
    # if mode == "coarse":
    #     # Coarse mode: 256x128 粒度
    #     def flex_mask_mod(b, h, q_idx, kv_idx):
    #         q_block_idx = q_idx // q_block_size
    #         k_block_idx = kv_idx // k_block_size
    #         return block_mask_binary[b, h, q_block_idx, k_block_idx].bool()
    # else:
        # Fine mode: 128x128 粒度
    flex_mask_mod = generate_upsampled_mask_mod(block_mask_binary, block_size=block_size)
    
    try:
        # 使用 _compile=True 来避免 materialize full mask (O(seq²) 内存)
        # 如果 mask 在所有 batch 和 head 上相同，可以用 B=None, H=None 来 broadcast
        block_mask_flex = create_block_mask(
            flex_mask_mod,
            B=batch_size,
            H=num_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            BLOCK_SIZE=(block_size, block_size),
            _compile=True,  # 关键：启用编译模式避免 full mask materialization
        )
        flex_mask_created = True
        print(f"FlexAttention block mask created successfully (_compile=True)")
    except Exception as e:
        print(colored(f"Warning: Failed to create FlexAttention block mask: {e}", "yellow"))
        import traceback
        traceback.print_exc()
        block_mask_flex = None
        flex_mask_created = False
    
    # 转换为 FA4 格式
    # 两种 mode 都不需要 mask_mod，完全依赖 block_sparse_tensors
    block_sparse_tensors = None
    
    if FA4_AVAILABLE:
        try:
            # 对于 SM100，FA4 内部会自动设置 q_stage=2（当 seqlen > m_block_size）
            # block_sparse_tensors 的 num_q_blocks 应该是以有效粒度计算的
            # 有效 Q 粒度 = q_stage * m_block_size
            # - Coarse: m_block_size=128, 有效 Q 粒度=256 -> mask 是 256 粒度，q_stage=1 给 convert
            # - Fine: m_block_size=64, 有效 Q 粒度=128 -> mask 是 128 粒度，q_stage=1 给 convert
            # 注：q_stage 参数在 convert_binary_mask_to_fa4_format 中只用于合并相邻 Q blocks
            # 当 mask 的粒度已经和有效 Q 粒度匹配时，不需要合并
            block_sparse_tensors = convert_binary_mask_to_fa4_format(
                block_mask_binary, q_stage=1  # mask 粒度已经匹配有效 Q 粒度，不需要合并
            )
        except Exception as e:
            print(colored(f"Warning: Failed to convert to FA4 format: {e}", "yellow"))
            import traceback
            traceback.print_exc()
    
    # === 精度测试 ===
    if run_accuracy:
        print(colored("\n--- Accuracy Test ---", "yellow"))
        
        # PyTorch Reference - 使用 chunked 版本，动态生成 mask 避免 OOM
        # 对于大序列（如 130K），完整 expanded_mask 需要约 1TB，会 OOM
        # chunked 版本直接使用 block_mask，每次只扩展当前 chunk 的 mask
        try:
            # 优先使用 chunked 版本，更节省内存
            ref_output = pytorch_reference_chunked(
                query, key, value,
                block_mask=block_mask_binary,
                q_block_size=q_block_size,
                k_block_size=k_block_size,
                chunk_size=1024,  # 每次处理 1024 个 Q tokens
            )
        except Exception as e:
            print(colored(f"Error in PyTorch reference: {e}", "red"))
            import traceback
            traceback.print_exc()
            ref_output = None
        
        # FA4 Block Sparse
        if FA4_AVAILABLE and block_sparse_tensors is not None and ref_output is not None:
            try:
                print(query.shape)
                print(block_sparse_tensors.mask_block_cnt.shape)
                print(block_sparse_tensors.mask_block_idx.shape)
                fa4_output = fa4_sparse_attention(
                    query, key, value,
                    block_sparse_tensors=block_sparse_tensors,
                    m_block_size=m_block_size,
                    n_block_size=n_block_size,
                )
                validate_accuracy(
                    ref_output, fa4_output,
                    message=f"PyTorch Reference vs FA4 Block Sparse ({mode} mode)"
                )
            except Exception as e:
                print(colored(f"FA4 Error: {e}", "red"))
                import traceback
                traceback.print_exc()
        elif not FA4_AVAILABLE:
            print(colored("⚠️ FA4 not available, skipping FA4 accuracy test", "yellow"))


        # FlexAttention - 使用 compiled 版本避免 materialize full scores matrix
        if flex_mask_created and ref_output is not None:
            try:
                flex_output = flex_sparse_attention(
                    query, key, value, block_mask_flex, flex_attention_compiled
                )
                validate_accuracy(
                    ref_output, flex_output,
                    message="PyTorch Reference vs FlexAttention"
                )
            except Exception as e:
                print(colored(f"FlexAttention Error: {e}", "red"))
                import traceback
                traceback.print_exc()

    
    # === 性能测试 ===
    summary: Optional[dict[str, str]] = None
    if run_performance:
        print(colored("\n--- Performance Test ---", "yellow"))
        
        results = {}
        mode_label = f"FA4 Block Sparse ({mode})"
        
        # Warmup
        print(colored(f"Warming up ({WARMUP_ITERATIONS} iterations)...", "magenta"))
        for _ in range(WARMUP_ITERATIONS):
            _ = pytorch_sdpa(query, key, value)
            if flex_mask_created:
                try:
                    _ = flex_sparse_attention(
                        query, key, value, block_mask_flex, flex_attention_compiled
                    )
                except Exception:
                    pass
            if FA4_AVAILABLE and block_sparse_tensors is not None:
                try:
                    _ = fa4_sparse_attention(
                        query, key, value,
                        block_sparse_tensors=block_sparse_tensors,
                        m_block_size=m_block_size,
                        n_block_size=n_block_size,
                    )
                except Exception:
                    pass
        torch.cuda.synchronize()
        
        # PyTorch SDPA (dense)
        print(colored(
            f"Running PyTorch SDPA ({PERF_BENCHMARK_ITERATIONS} iterations)...",
            "magenta"
        ))
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with nvtx.annotate("Perf.SDPA", color="red"):
            start_event.record()
            for _ in range(PERF_BENCHMARK_ITERATIONS):
                _ = pytorch_sdpa(query, key, value)
            end_event.record()
        torch.cuda.synchronize()
        results['PyTorch SDPA (dense)'] = (
            start_event.elapsed_time(end_event) / PERF_BENCHMARK_ITERATIONS
        )
        
        # FlexAttention
        if flex_mask_created:
            print(colored(
                f"Running FlexAttention ({PERF_BENCHMARK_ITERATIONS} iterations)...",
                "magenta"
            ))
            try:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                with nvtx.annotate("Perf.FlexAttn", color="green"):
                    start_event.record()
                    for _ in range(PERF_BENCHMARK_ITERATIONS):
                        _ = flex_sparse_attention(
                            query, key, value, block_mask_flex, flex_attention_compiled
                        )
                    end_event.record()
                torch.cuda.synchronize()
                results['FlexAttention (sparse)'] = (
                    start_event.elapsed_time(end_event) / PERF_BENCHMARK_ITERATIONS
                )
            except Exception as e:
                print(colored(f"FlexAttention benchmark error: {e}", "red"))
                results['FlexAttention (sparse)'] = float('nan')
        
        # FA4 Block Sparse
        if FA4_AVAILABLE and block_sparse_tensors is not None:
            print(colored(
                f"Running {mode_label} ({PERF_BENCHMARK_ITERATIONS} iterations)...",
                "magenta"
            ))
            try:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                with nvtx.annotate("Perf.FA4", color="blue"):
                    start_event.record()
                    for _ in range(PERF_BENCHMARK_ITERATIONS):
                        _ = fa4_sparse_attention(
                            query, key, value,
                            block_sparse_tensors=block_sparse_tensors,
                            m_block_size=m_block_size,
                            n_block_size=n_block_size,
                        )
                    end_event.record()
                torch.cuda.synchronize()
                results[mode_label] = (
                    start_event.elapsed_time(end_event) / PERF_BENCHMARK_ITERATIONS
                )
            except Exception as e:
                print(colored(f"FA4 benchmark error: {e}", "red"))
                import traceback
                traceback.print_exc()
                results[mode_label] = float('nan')
        elif not FA4_AVAILABLE:
            print(colored("⚠️ FA4 not available, skipping FA4 performance test", "yellow"))
        
        # 打印结果
        print("\n🔥 Performance Results:")
        base_time = results.get('PyTorch SDPA (dense)', 1.0)
        for name, time in results.items():
            if math.isnan(time):
                print(f"  {name}: ERROR")
            else:
                speedup = base_time / time if time > 0 else 0
                print(f"  {name}: {time:.4f} ms (speedup vs SDPA: {speedup:.2f}x)")

        if base_time is not None and not math.isnan(base_time):
            summary = {
                "Sparsity": f"{sparsity * 100:.0f}%",
                "Shape": f"({batch_size},{seq_len},{num_heads},{head_dim})",
                "Mode": mode,
                "SDPA (ms)": f"{base_time:.4f}",
            }
            flex_time = results.get('FlexAttention (sparse)')
            if flex_time is not None and not math.isnan(flex_time):
                summary["FlexAttn (ms)"] = f"{flex_time:.4f}"
                summary["FlexAttn Speedup"] = f"{base_time / flex_time:.2f}x"
            fa4_time = results.get(mode_label)
            if fa4_time is not None and not math.isnan(fa4_time):
                summary["FA4 (ms)"] = f"{fa4_time:.4f}"
                summary["FA4 Speedup"] = f"{base_time / fa4_time:.2f}x"
    
    torch.cuda.empty_cache()
    print("")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FA4 Block Sparse Attention Demo for B200 (Blackwell/SM100)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to run on"
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5,
        help="Sparsity level (0-1, higher = more sparse)"
    )
    parser.add_argument(
        "--accuracy-only", action="store_true",
        help="Only run accuracy tests"
    )
    parser.add_argument(
        "--performance-only", action="store_true",
        help="Only run performance tests"
    )
    parser.add_argument(
        "--log", action="store_true",
        help="Enable logging to file"
    )
    parser.add_argument(
        "--block-size", type=int, default=128,
        help="Block size for sparse attention (default: 128)"
    )
    parser.add_argument(
        "--mode", type=str, default="coarse", choices=["coarse", "fine"],
        help="Sparse mode: 'coarse' (256x128, no mask_mod) or 'fine' (128x128 + CuTe mask_mod)"
    )
    args = parser.parse_args()
    
    # 设置日志
    tee_logger = None
    if args.log:
        log_filename = datetime.now().strftime("fa4_demo_%Y%m%d_%H%M%S.log")
        tee_logger = TeeLogger(log_filename)
        sys.stdout = tee_logger
        print(f"日志将保存到: {log_filename}")
    
    device = torch.device(args.device)
    mode_desc = "coarse (256x128)" if args.mode == "coarse" else "fine (128x128 + mask_mod)"
    print(colored(f"Running FA4 Block Sparse Demo on device: {device}", "cyan"))
    print(f"Mode: {mode_desc}")
    print(f"FA4 Available: {FA4_AVAILABLE}")
    print(f"CUDA Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    print("")
    
    run_accuracy = not args.performance_only
    run_performance = not args.accuracy_only
    perf_results: list[dict[str, str]] = []
    
    # 定义测试 case: (batch_size, seq_len, num_heads, head_dim)
    test_cases = [
        # head_dim = 64
        (8, 130560, 4, 64),      # Case 1: 大序列长度
        (64, 65280, 2, 64),      # Case 2: 大 batch
        # head_dim = 128
        (2, 1590, 4, 128),       # Case 3: 小序列
        (2, 8160, 4, 128),       # Case 4: 中等序列
    ]
    
    # 测试多个稀疏度
    sparsity_levels = [0.1, 0.4, 0.8]  # 10%, 40%, 80%
    
    test_idx = 0
    for sparsity in sparsity_levels:
        print(colored(f"\n{'#' * 80}", "magenta"))
        print(colored(f"# Testing Sparsity: {sparsity * 100:.0f}%", "magenta"))
        print(colored(f"{'#' * 80}\n", "magenta"))
        
        for batch_size, seq_len, num_heads, head_dim in test_cases:
            test_idx += 1
            # 使用 chunked reference，所有 case 都可以做精度测试
            case_run_accuracy = run_accuracy
            # print case and sparsity
            print(colored(f"Test Case {test_idx}: B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}, Sparsity: {sparsity * 100:.0f}%", "cyan"))
            try:
                summary = run_single_test(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    block_size=args.block_size,
                    sparsity=sparsity,
                    device=args.device,
                    run_accuracy=case_run_accuracy,
                    run_performance=run_performance,
                    test_case_idx=test_idx,
                )
                if summary is not None:
                    perf_results.append(summary)
            except Exception as e:
                print(colored(f"Test case {test_idx} failed: {e}", "red"))
                import traceback
                traceback.print_exc()
                continue

    if run_performance and perf_results:
        print(colored("\n" + "=" * 100, "green"))
        print(colored("Performance Summary Table", "green", attrs=["bold"]))
        print(colored("=" * 100, "green"))
        df = pd.DataFrame(perf_results)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.colheader_justify", "center")
        print(df.to_string(index=False))
        print("")
    
    if tee_logger:
        tee_logger.close()
        sys.stdout = tee_logger.terminal


if __name__ == "__main__":
    main()
