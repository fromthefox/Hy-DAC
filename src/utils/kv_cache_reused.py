"""KV-Cache reuse vs. full r.compute timing de.o usin(..:al LLaMA wei,hts."""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL_PATH = "/Users/yhbian/Downloads/Models/Llama-3.2-3B"
PARAMS_PATH = "/Users/yhbian/Downloads/Models/Llama-3.2-3B/params.json"

NUM_DEVICES = 4
OFFLINE_DEVICE = 1
OFFLINE_AT_STEP = 5
LAYERS_TO_SIMULATE = 28
MAX_SEQ_LEN = 128


# ==================== 数据结构 ====================
@dataclass
class DeviceState:
    """设备状态"""
    device_id: int
    head_range: Tuple[int, int]  # (start_head, end_head) 左闭右开
    is_online: bool = True
    kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None  # [(k, v)] for each layer


@dataclass
class ModelConfig:
    """模型配置"""
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    norm_eps: float
    rope_theta: float = 500000.0
    ffn_dim_multiplier: float = 4.0
    multiple_of: int = 256
    use_scaled_rope: bool = False
    
    @property
    def head_dim(self):
        return self.dim // self.n_heads


def load_model_config(params_path: str) -> ModelConfig:
    params = json.loads(Path(params_path).read_text())
    return ModelConfig(
        dim=params["dim"],
        n_layers=params["n_layers"],
        n_heads=params["n_heads"],
        n_kv_heads=params.get("n_kv_heads", params["n_heads"]),
        vocab_size=params["vocab_size"],
        norm_eps=params["norm_eps"],
        rope_theta=params.get("rope_theta", 500000.0),
        ffn_dim_multiplier=params.get("ffn_dim_multiplier", 4.0),
        multiple_of=params.get("multiple_of", 256),
    )


def load_model_weights(model_dir: str) -> Dict[str, torch.Tensor]:
    """支持加载分片的 safetensors 文件"""
    weights: Dict[str, torch.Tensor] = {}
    
    # 查找所有 safetensors 文件
    model_path = Path(model_dir)
    safetensor_files = sorted(model_path.glob("model-*.safetensors"))
    
    if not safetensor_files:
        raise FileNotFoundError(f"未在 {model_dir} 找到 safetensors 文件")
    
    print(f"找到 {len(safetensor_files)} 个权重文件:")
    for f in safetensor_files:
        print(f"  - {f.name}")
    
    # 逐个加载每个分片
    for safetensor_file in safetensor_files:
        print(f"加载 {safetensor_file.name}...")
        with safe_open(str(safetensor_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    
    print(f"共加载 {len(weights)} 个权重张量")
    return weights

# 新增: 通用嵌入权重解析
def get_embedding_weight(weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    candidates = [
        "tok_embeddings.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "model.tok_embeddings.weight",
        "transformer.wte.weight"
    ]
    for name in candidates:
        if name in weights:
            return weights[name]
    # 兜底启发式
    for k in weights.keys():
        if "embed" in k and k.endswith("weight"):
            print(f"⚠️ 使用推测嵌入权重: {k}")
            return weights[k]
    raise KeyError(
        "未找到词嵌入权重，尝试候选: "
        + ", ".join(candidates)
        + "\n可用键示例: "
        + ", ".join(list(weights.keys())[:30])
        + " ..."
    )

# 新增: 通用注意力层权重解析
def get_attn_weights(weights: Dict[str, torch.Tensor], layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回 (wq, wk, wv, wo)
    支持多种模型命名格式:
      - llama.cpp: layers.{i}.attention.wq.weight
      - HF Llama: model.layers.{i}.self_attn.q_proj.weight
      - 通用 transformer: transformer.h.{i}.attn.q_proj.weight
    """
    patterns = [
        (
            f"layers.{layer_idx}.attention.wq.weight",
            f"layers.{layer_idx}.attention.wk.weight",
            f"layers.{layer_idx}.attention.wv.weight",
            f"layers.{layer_idx}.attention.wo.weight",
        ),
        (
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        ),
        (
            f"transformer.h.{layer_idx}.attn.q_proj.weight",
            f"transformer.h.{layer_idx}.attn.k_proj.weight",
            f"transformer.h.{layer_idx}.attn.v_proj.weight",
            f"transformer.h.{layer_idx}.attn.o_proj.weight",
        ),
    ]
    for qkvo in patterns:
        if all(k in weights for k in qkvo):
            return tuple(weights[k] for k in qkvo)
    # 兜底: 尝试根据包含层索引和 q_proj/k_proj 等关键词自动匹配
    keys_layer = [k for k in weights.keys() if f".{layer_idx}." in k]
    q = next((k for k in keys_layer if "q_proj.weight" in k or "wq.weight" in k), None)
    k = next((k for k in keys_layer if "k_proj.weight" in k or "wk.weight" in k), None)
    v = next((k for k in keys_layer if "v_proj.weight" in k or "wv.weight" in k), None)
    o = next((k for k in keys_layer if "o_proj.weight" in k or "wo.weight" in k), None)
    if all([q, k, v, o]):
        print(f"⚠️ 自动匹配注意力权重: {q}, {k}, {v}, {o}")
        return weights[q], weights[k], weights[v], weights[o]
    raise KeyError(
        f"未找到第 {layer_idx} 层注意力权重。已尝试模式: "
        + "; ".join("|".join(p) for p in patterns)
        + "\n可用相关键示例: "
        + ", ".join(keys_layer[:20])
    )


def get_norm_weight(weights: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    candidates = ["norm.weight", "model.norm.weight", "transformer.ln_f.weight"]
    for key in candidates:
        if key in weights:
            return weights[key]
    for key in weights:
        if key.endswith("norm.weight") or key.endswith("ln_f.weight"):
            print(f"⚠️ 使用推测归一化权重: {key}")
            return weights[key]
    # print("⚠️ 未找到归一化权重，使用无权重 layer_norm。")
    return None


def get_output_weight(weights: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    candidates = ["output.weight", "lm_head.weight", "model.lm_head.weight"]
    for key in candidates:
        if key in weights:
            return weights[key]
    for key in weights:
        if key.endswith("lm_head.weight") or ("output" in key and key.endswith(".weight")):
            print(f"⚠️ 使用推测输出权重: {key}")
            return weights[key]
    # print("⚠️ 未找到输出层权重，跳过 logits 计算。")
    return None

def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(end)
    freqs = torch.outer(positions, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 这一部分的原理可以参考Llama3-from-scratch项目的解释
    head_dim = xq.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for rotary embeddings")
    seq_len = xq.shape[1]
    freqs = freqs_cis[:seq_len].view(1, seq_len, 1, -1)
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    rotated_q = torch.view_as_real(xq_complex * freqs).reshape_as(xq)
    rotated_k = torch.view_as_real(xk_complex * freqs).reshape_as(xk)
    return rotated_q.type_as(xq), rotated_k.type_as(xk)


def select_kv_slice(matrix: torch.Tensor, start_head: int, end_head: int, config: ModelConfig) -> torch.Tensor:
    group = max(1, config.n_heads // config.n_kv_heads)
    if matrix.shape[0] != config.n_kv_heads * config.head_dim:
        raise ValueError(
            f"KV 权重形状不符: 期望 {(config.n_kv_heads * config.head_dim, config.dim)}, 实际 {tuple(matrix.shape)}"
        )
    kv_weights = matrix.view(config.n_kv_heads, config.head_dim, -1)
    kv_indices = (torch.arange(start_head, end_head) // group).tolist()
    selected = kv_weights[kv_indices]  # [num_heads, head_dim, dim]
    return selected.reshape(-1, kv_weights.shape[-1])


def compute_attention_partial_heads(
    x: torch.Tensor,
    layer_idx: int,
    weights: Dict[str, torch.Tensor],
    config: ModelConfig,
    head_range: Tuple[int, int],
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    batch, seq_len, _ = x.shape
    start_head, end_head = head_range
    num_heads = end_head - start_head
    head_dim = config.head_dim

    wq, wk, wv, _ = get_attn_weights(weights, layer_idx)
    q = F.linear(x, wq[start_head * head_dim : end_head * head_dim, :])
    k = F.linear(x, select_kv_slice(wk, start_head, end_head, config))
    v = F.linear(x, select_kv_slice(wv, start_head, end_head, config))

    q = q.view(batch, seq_len, num_heads, head_dim)
    k = k.view(batch, seq_len, num_heads, head_dim)
    v = v.view(batch, seq_len, num_heads, head_dim)

    q, k = apply_rotary_emb(q, k, freqs_cis[positions])

    if kv_cache is not None:
        past_k, past_v = kv_cache
        k = torch.cat([past_k, k], dim=1)
        v = torch.cat([past_v, v], dim=1)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v)
    output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

    return output, (k.transpose(1, 2), v.transpose(1, 2))


def initialize_devices(config: ModelConfig, num_devices: int) -> List[DeviceState]:
    heads_per_device = config.n_heads // num_devices
    devices: List[DeviceState] = []
    for idx in range(num_devices):
        start = idx * heads_per_device
        end = (idx + 1) * heads_per_device if idx < num_devices - 1 else config.n_heads
        devices.append(DeviceState(idx, (start, end), True, [None] * config.n_layers))
    return devices


def reassign_heads(devices: List[DeviceState], offline_device_id: int) -> Dict[int, List[Tuple[int, int]]]:
    offline = devices[offline_device_id]
    start, end = offline.head_range
    online = [d for d in devices if d.is_online and d.device_id != offline_device_id]
    if not online:
        return {}
    num_lost = end - start
    per_device = num_lost // len(online)
    remainder = num_lost % len(online)
    assignments: Dict[int, List[Tuple[int, int]]] = {}
    cursor = start
    for idx, device in enumerate(online):
        take = per_device + (1 if idx < remainder else 0)
        if take == 0:
            continue
        assignments[device.device_id] = [(cursor, cursor + take)]
        cursor += take
    return assignments


def run_layer(
    devices: List[DeviceState],
    layer_idx: int,
    x: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    config: ModelConfig,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    extra_assignments: Optional[Dict[int, List[Tuple[int, int]]]],
    read_cache: bool,
    write_cache: bool,
) -> torch.Tensor:
    outputs: List[Optional[torch.Tensor]] = [None] * config.n_heads
    head_dim = config.head_dim

    for device in devices:
        if not device.is_online:
            continue
        head_ranges = [device.head_range]
        if extra_assignments and device.device_id in extra_assignments:
            head_ranges.extend(extra_assignments[device.device_id])

        for head_range in head_ranges:
            start_head, end_head = head_range
            kv_cache = device.kv_cache[layer_idx] if (read_cache and head_range == device.head_range) else None

            attn_out, new_kv = compute_attention_partial_heads(
                x,
                layer_idx,
                weights,
                config,
                head_range,
                kv_cache,
                freqs_cis,
                positions,
            )

            if write_cache and head_range == device.head_range:
                device.kv_cache[layer_idx] = new_kv

            reshaped = attn_out.view(1, -1, end_head - start_head, head_dim)
            for local_idx, head_idx in enumerate(range(start_head, end_head)):
                outputs[head_idx] = reshaped[:, :, local_idx, :]

    concat = torch.cat(outputs, dim=-1).view_as(x)
    _, _, _, wo = get_attn_weights(weights, layer_idx)
    return x + F.linear(concat, wo)


def layer_norm_and_output(x: torch.Tensor, weights: Dict[str, torch.Tensor], dim: int) -> torch.Tensor:
    norm_weight = get_norm_weight(weights)
    output_weight = get_output_weight(weights)
    x = F.layer_norm(x, (dim,), norm_weight)
    return F.linear(x, output_weight) if output_weight is not None else x


def prepare_inputs(tokens: List[int], embed_weight: torch.Tensor, read_cache: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if read_cache and len(tokens) > 1:
        input_ids = torch.tensor([[tokens[-1]]], dtype=torch.long)
        positions = torch.tensor([len(tokens) - 1], dtype=torch.long)
    else:
        input_ids = torch.tensor([tokens], dtype=torch.long)
        positions = torch.arange(len(tokens), dtype=torch.long)
    x = F.embedding(input_ids, embed_weight)
    return x, positions


def run_step(
    devices: List[DeviceState],
    tokens: List[int],
    weights: Dict[str, torch.Tensor],
    config: ModelConfig,
    freqs_cis: torch.Tensor,
    embed_weight: torch.Tensor,
    extra_assignments: Optional[Dict[int, List[Tuple[int, int]]]],
    read_cache: bool,
    write_cache: bool,
) -> float:
    x, positions = prepare_inputs(tokens, embed_weight, read_cache)
    start = time.perf_counter()
    for layer_idx in range(min(config.n_layers, LAYERS_TO_SIMULATE)):
        x = run_layer(
            devices,
            layer_idx,
            x,
            weights,
            config,
            freqs_cis,
            positions,
            extra_assignments,
            read_cache,
            write_cache,
        )
    _ = layer_norm_and_output(x, weights, config.dim)
    return time.perf_counter() - start


def build_prompt_tokens(config: ModelConfig) -> List[int]:
    bos = min(config.vocab_size - 1, 128000)
    base = [bos, 172, 201, 208, 208]
    return [token % config.vocab_size for token in base]


def print_weight_evidence(weights: Dict[str, torch.Tensor]) -> None:
    sample_key = next(iter(weights))
    tensor = weights[sample_key].float()
    print(
        f"  权重样本 [{sample_key}] -> 形状 {tuple(tensor.shape)}, "
        f"均值 {tensor.mean():.6f}, 标准差 {tensor.std():.6f}, |max| {tensor.abs().max():.6f}"
    )


def main() -> None:
    torch.set_grad_enabled(False)

    print("=" * 60)
    print("KV-Cache Reuse Demo (Real Weights)")
    print("=" * 60)

    config = load_model_config(PARAMS_PATH)
    weights = load_model_weights(MODEL_PATH)  # ← 传入目录而不是文件
    embed_weight = get_embedding_weight(weights)
    print(f"模型维度: {config.dim}, 层数: {config.n_layers}, 注意力头: {config.n_heads}, KV 头: {config.n_kv_heads}")
    print_weight_evidence(weights)

    devices_reuse = initialize_devices(config, NUM_DEVICES)
    devices_baseline = initialize_devices(config, NUM_DEVICES)
    freqs_cis = precompute_freqs_cis(config.head_dim, MAX_SEQ_LEN, config.rope_theta)

    tokens = build_prompt_tokens(config)
    reuse_times: List[float] = []
    baseline_times: List[float] = []
    reuse_assignments: Optional[Dict[int, List[Tuple[int, int]]]] = None
    baseline_assignments: Optional[Dict[int, List[Tuple[int, int]]]] = None
    baseline_needs_full_recompute = False

    for step in range(10):
        print(f"\n-- Step {step + 1} --")

        if step == OFFLINE_AT_STEP:
            devices_reuse[OFFLINE_DEVICE].is_online = False
            reuse_assignments = reassign_heads(devices_reuse, OFFLINE_DEVICE)
            print(f"设备 {OFFLINE_DEVICE} 下线，重新分配头: {reuse_assignments}")

            devices_baseline = initialize_devices(config, NUM_DEVICES - 1)
            baseline_assignments = None
            baseline_needs_full_recompute = True
        
        # 复用 KV-Cache 路径
        reuse_elapsed = run_step(
            devices_reuse,
            tokens,
            weights,
            config,
            freqs_cis,
            embed_weight,
            reuse_assignments,
            read_cache=True,
            write_cache=True,
        )
        
        # 基线路径：掉线那步强制不使用缓存，之后才恢复
        baseline_read_cache = not baseline_needs_full_recompute
        baseline_elapsed = run_step(
            devices_baseline,
            tokens,
            weights,
            config,
            freqs_cis,
            embed_weight,
            baseline_assignments,
            read_cache=baseline_read_cache,
            write_cache=True,
        )
        
        baseline_needs_full_recompute = False

        speedup = baseline_elapsed / reuse_elapsed if reuse_elapsed > 0 else float("inf")
        print(f"复用 KV-Cache: {reuse_elapsed:.4f}s | 全量重算: {baseline_elapsed:.4f}s | 加速比: {speedup:.2f}x")

        tokens.append((tokens[-1] + 1) % config.vocab_size)

        # 修复：记录每步耗时
        reuse_times.append(reuse_elapsed)
        baseline_times.append(baseline_elapsed)
    
    total_reuse = sum(reuse_times)
    total_baseline = sum(baseline_times)
    avg_reuse = total_reuse / len(reuse_times)
    avg_baseline = total_baseline / len(baseline_times)
    overall = total_baseline / total_reuse if total_reuse > 0 else float("inf")

    print("\n" + "-" * 60)
    print("Timing Summary")
    print("-" * 60)
    print(f"使用 KV-Cache: 总耗时 {total_reuse:.4f}s | 平均 {avg_reuse:.4f}s")
    print(f"全量重算   : 总耗时 {total_baseline:.4f}s | 平均 {avg_baseline:.4f}s")
    print(f"整体加速比 : {overall:.2f}x")


if __name__ == "__main__":
    main()