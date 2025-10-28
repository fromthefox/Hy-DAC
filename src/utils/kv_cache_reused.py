"""
分布式推理 KV-Cache 容错 Demo
模拟场景：4个设备分别持有不同的注意力头，某个设备下线后其他设备接管并复用现有 KV-Cache
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
from dataclasses import dataclass
from safetensors import safe_open

# ==================== 配置部分 ====================
MODEL_PATH = "A"  # 替换为你的 model.safetensors 路径
PARAMS_PATH = "B"  # 替换为你的 params.json 路径
TOKENIZER_PATH = "C"  # 替换为你的 tokenizer.model 路径

NUM_DEVICES = 4  # 模拟4个设备
OFFLINE_DEVICE = 1  # 模拟第2个设备下线（索引从0开始）
OFFLINE_AT_TOKEN = 5  # 在生成第5个token后设备下线


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
    
    @property
    def head_dim(self):
        return self.dim // self.n_heads


# ==================== 简化的 Tokenizer ====================
class SimpleTokenizer:
    """简化的 Tokenizer（用于 Demo）"""
    def __init__(self, tokenizer_path: str):
        # 这里简化处理，实际应该加载 sentencepiece
        self.bos_id = 128000
        self.eos_id = 128001
        
    def encode(self, text: str) -> List[int]:
        """简化编码：为 Demo 生成假的 token ids"""
        # 实际使用时应该用 sentencepiece 或 transformers
        return [self.bos_id] + [ord(c) % 1000 + 100 for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """简化解码"""
        return f"<generated_{len(tokens)}_tokens>"


# ==================== 模型加载 ====================
def load_model_config(params_path: str) -> ModelConfig:
    """加载模型配置"""
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    return ModelConfig(
        dim=params['dim'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
        n_kv_heads=params.get('n_kv_heads', params['n_heads']),
        vocab_size=params['vocab_size'],
        norm_eps=params['norm_eps'],
        rope_theta=params.get('rope_theta', 500000.0)
    )


def load_model_weights(model_path: str) -> Dict[str, torch.Tensor]:
    """加载模型权重"""
    weights = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


# ==================== RoPE 实现 ====================
def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    """预计算 RoPE 的频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """应用 RoPE"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis[:xq_.shape[1]]
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==================== 注意力计算（支持部分头） ====================
def compute_attention_partial_heads(
    x: torch.Tensor,
    layer_idx: int,
    weights: Dict[str, torch.Tensor],
    config: ModelConfig,
    head_range: Tuple[int, int],
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    计算部分注意力头
    
    Args:
        x: 输入 [batch, seq_len, dim]
        layer_idx: 层索引
        weights: 模型权重
        config: 模型配置
        head_range: (start_head, end_head) 要计算的头范围
        kv_cache: 已有的 KV-Cache (k, v)
        freqs_cis: RoPE 频率
        positions: 位置索引
    
    Returns:
        attention_output: [batch, seq_len, head_dim * num_heads_in_range]
        new_kv_cache: 更新后的 (k, v)
    """
    batch, seq_len, dim = x.shape
    start_head, end_head = head_range
    num_heads_in_range = end_head - start_head
    head_dim = config.head_dim
    
    # 加载权重
    wq = weights[f'layers.{layer_idx}.attention.wq.weight']
    wk = weights[f'layers.{layer_idx}.attention.wk.weight']
    wv = weights[f'layers.{layer_idx}.attention.wv.weight']
    
    # 只取对应头的权重切片
    head_start_dim = start_head * head_dim
    head_end_dim = end_head * head_dim
    
    wq_slice = wq[head_start_dim:head_end_dim, :]
    wk_slice = wk[head_start_dim:head_end_dim, :]
    wv_slice = wv[head_start_dim:head_end_dim, :]
    
    # 计算 Q, K, V
    q = F.linear(x, wq_slice)  # [batch, seq_len, num_heads_in_range * head_dim]
    k = F.linear(x, wk_slice)
    v = F.linear(x, wv_slice)
    
    # Reshape
    q = q.view(batch, seq_len, num_heads_in_range, head_dim)
    k = k.view(batch, seq_len, num_heads_in_range, head_dim)
    v = v.view(batch, seq_len, num_heads_in_range, head_dim)
    
    # 应用 RoPE
    q, k = apply_rotary_emb(q, k, freqs_cis[positions])
    
    # 更新 KV-Cache
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        k = torch.cat([k_cache, k], dim=1)
        v = torch.cat([v_cache, v], dim=1)
    
    # 计算注意力
    q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    scores = torch.matmul(q, k.transpose(2, 3)) / (head_dim ** 0.5)
    scores = F.softmax(scores, dim=-1)
    output = torch.matmul(scores, v)  # [batch, num_heads, seq_len, head_dim]
    
    # Reshape
    output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    
    return output, (k.transpose(1, 2), v.transpose(1, 2))


# ==================== 设备管理 ====================
def initialize_devices(config: ModelConfig, num_devices: int) -> List[DeviceState]:
    """初始化设备，分配注意力头"""
    heads_per_device = config.n_heads // num_devices
    devices = []
    
    for i in range(num_devices):
        start_head = i * heads_per_device
        end_head = (i + 1) * heads_per_device if i < num_devices - 1 else config.n_heads
        
        device = DeviceState(
            device_id=i,
            head_range=(start_head, end_head),
            kv_cache=[None] * config.n_layers  # 每层一个 KV-Cache
        )
        devices.append(device)
        print(f"设备 {i}: 负责头 [{start_head}, {end_head})")
    
    return devices


def reassign_heads_after_failure(
    devices: List[DeviceState],
    offline_device_id: int,
    config: ModelConfig
) -> Dict[int, List[Tuple[int, int]]]:
    """
    设备下线后重新分配头
    
    返回: {device_id: [(start_head, end_head), ...]} 额外分配的头
    """
    offline_device = devices[offline_device_id]
    offline_heads = offline_device.head_range
    
    print(f"\n⚠️  设备 {offline_device_id} 下线，丢失头 {offline_heads}")
    
    # 简单策略：将丢失的头平均分配给其他在线设备
    online_devices = [d for d in devices if d.is_online and d.device_id != offline_device_id]
    num_lost_heads = offline_heads[1] - offline_heads[0]
    heads_per_online_device = num_lost_heads // len(online_devices)
    
    reassignment = {}
    current_head = offline_heads[0]
    
    for i, device in enumerate(online_devices):
        extra_heads = heads_per_online_device
        if i == len(online_devices) - 1:  # 最后一个设备承担剩余的
            extra_heads = offline_heads[1] - current_head
        
        reassignment[device.device_id] = [(current_head, current_head + extra_heads)]
        print(f"  → 设备 {device.device_id} 额外承担头 [{current_head}, {current_head + extra_heads})")
        current_head += extra_heads
    
    return reassignment


# ==================== 推理主流程 ====================
def generate_token(
    x: torch.Tensor,
    devices: List[DeviceState],
    weights: Dict[str, torch.Tensor],
    config: ModelConfig,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    extra_assignments: Optional[Dict[int, List[Tuple[int, int]]]] = None
) -> torch.Tensor:
    """
    生成一个 token
    
    Args:
        x: 输入 embeddings [batch, seq_len, dim]
        devices: 设备列表
        weights: 模型权重
        config: 模型配置
        freqs_cis: RoPE 频率
        positions: 位置索引
        extra_assignments: 设备下线后的额外头分配
    
    Returns:
        logits: [batch, seq_len, vocab_size]
    """
    batch, seq_len, dim = x.shape
    
    # 逐层计算
    for layer_idx in range(config.n_layers):
        # RMSNorm (简化：直接用 layer_norm)
        attn_input = x
        
        # 收集所有设备的注意力输出
        all_head_outputs = [None] * config.n_heads
        
        for device in devices:
            if not device.is_online:
                continue
            
            # 计算设备原本负责的头
            head_ranges = [device.head_range]
            
            # 如果有额外分配的头（因为其他设备下线）
            if extra_assignments and device.device_id in extra_assignments:
                head_ranges.extend(extra_assignments[device.device_id])
            
            for head_range in head_ranges:
                start_head, end_head = head_range
                
                # 检查是否有可复用的 KV-Cache
                kv_cache = device.kv_cache[layer_idx]
                
                # 如果是额外分配的头（来自下线设备），需要重新计算
                if head_range != device.head_range:
                    print(f"  🔄 设备 {device.device_id} 重新计算头 [{start_head}, {end_head}) (Layer {layer_idx})")
                    kv_cache = None  # 强制重新计算
                else:
                    if kv_cache is not None:
                        print(f"  ✅ 设备 {device.device_id} 复用 KV-Cache 头 [{start_head}, {end_head}) (Layer {layer_idx})")
                
                # 计算注意力
                attn_out, new_kv = compute_attention_partial_heads(
                    attn_input,
                    layer_idx,
                    weights,
                    config,
                    head_range,
                    kv_cache,
                    freqs_cis,
                    positions
                )
                
                # 更新 KV-Cache（如果是原设备的头）
                if head_range == device.head_range:
                    device.kv_cache[layer_idx] = new_kv
                
                # 存储输出到对应的头位置
                num_heads_in_range = end_head - start_head
                head_dim = config.head_dim
                attn_out_reshaped = attn_out.view(batch, seq_len, num_heads_in_range, head_dim)
                
                for i, head_idx in enumerate(range(start_head, end_head)):
                    all_head_outputs[head_idx] = attn_out_reshaped[:, :, i, :]
        
        # 拼接所有头的输出
        concat_heads = torch.cat(all_head_outputs, dim=-1)  # [batch, seq_len, n_heads * head_dim]
        concat_heads = concat_heads.view(batch, seq_len, dim)
        
        # 输出投影
        wo = weights[f'layers.{layer_idx}.attention.wo.weight']
        attn_output = F.linear(concat_heads, wo)
        
        # 残差连接
        x = attn_input + attn_output
        
        # FFN (简化：跳过，只演示注意力部分)
        # 实际使用时需要完整实现
    
    # 输出层 (简化)
    norm_weight = weights['norm.weight']
    x = F.layer_norm(x, (dim,), norm_weight)
    
    output_weight = weights['output.weight']
    logits = F.linear(x, output_weight)
    
    return logits


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("分布式推理 KV-Cache 容错 Demo")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n📦 加载模型...")
    config = load_model_config(PARAMS_PATH)
    weights = load_model_weights(MODEL_PATH)
    tokenizer = SimpleTokenizer(TOKENIZER_PATH)
    
    print(f"  模型: Llama-3.2-1B")
    print(f"  层数: {config.n_layers}")
    print(f"  注意力头数: {config.n_heads}")
    print(f"  头维度: {config.head_dim}")
    
    # 2. 初始化设备
    print(f"\n🖥️  初始化 {NUM_DEVICES} 个设备...")
    devices = initialize_devices(config, NUM_DEVICES)
    
    # 3. 准备输入
    print("\n📝 准备输入...")
    prompt = "Hello"
    tokens = tokenizer.encode(prompt)[:10]  # 限制长度
    print(f"  输入 tokens: {tokens[:5]}...")
    
    # 预计算 RoPE
    max_seq_len = 128
    freqs_cis = precompute_freqs_cis(config.head_dim, max_seq_len, config.rope_theta)
    
    # 4. 开始生成
    print("\n🚀 开始生成...")
    num_tokens_to_generate = 10
    extra_assignments = None
    
    for step in range(num_tokens_to_generate):
        print(f"\n{'='*60}")
        print(f"生成第 {step + 1} 个 Token")
        print(f"{'='*60}")
        
        # 模拟设备下线
        if step == OFFLINE_AT_TOKEN:
            devices[OFFLINE_DEVICE].is_online = False
            extra_assignments = reassign_heads_after_failure(devices, OFFLINE_DEVICE, config)
        
        # 准备输入
        if step == 0:
            input_ids = torch.tensor([tokens], dtype=torch.long)
        else:
            input_ids = torch.tensor([[tokens[-1]]], dtype=torch.long)
        
        # Embedding
        embed_weight = weights['tok_embeddings.weight']
        x = F.embedding(input_ids, embed_weight)
        
        # 位置信息
        if step == 0:
            positions = torch.arange(len(tokens), dtype=torch.long)
        else:
            positions = torch.tensor([len(tokens) - 1], dtype=torch.long)
        
        # 生成 (只演示第一层)
        print("\n处理第 0 层:")
        start_time = time.time()
        
        # 简化版本：只计算一层来演示
        layer_idx = 0
        attn_input = x
        all_head_outputs = [None] * config.n_heads
        
        for device in devices:
            if not device.is_online:
                continue
            
            head_ranges = [device.head_range]
            if extra_assignments and device.device_id in extra_assignments:
                head_ranges.extend(extra_assignments[device.device_id])
            
            for head_range in head_ranges:
                start_head, end_head = head_range
                kv_cache = device.kv_cache[layer_idx]
                
                if head_range != device.head_range:
                    print(f"  🔄 设备 {device.device_id} 重新计算头 [{start_head}, {end_head})")
                    # 需要重新计算整个序列
                    if step > 0:
                        # 重新计算之前的所有 tokens
                        full_input_ids = torch.tensor([tokens], dtype=torch.long)
                        full_x = F.embedding(full_input_ids, embed_weight)
                        full_positions = torch.arange(len(tokens), dtype=torch.long)
                        
                        attn_out, new_kv = compute_attention_partial_heads(
                            full_x,
                            layer_idx,
                            weights,
                            config,
                            head_range,
                            None,  # 从头计算
                            freqs_cis,
                            full_positions
                        )
                        # 只取最后一个 token 的输出
                        attn_out = attn_out[:, -1:, :]
                    else:
                        attn_out, new_kv = compute_attention_partial_heads(
                            attn_input,
                            layer_idx,
                            weights,
                            config,
                            head_range,
                            None,
                            freqs_cis,
                            positions
                        )
                else:
                    if kv_cache is not None:
                        print(f"  ✅ 设备 {device.device_id} 复用 KV-Cache 头 [{start_head}, {end_head})")
                    
                    attn_out, new_kv = compute_attention_partial_heads(
                        attn_input,
                        layer_idx,
                        weights,
                        config,
                        head_range,
                        kv_cache,
                        freqs_cis,
                        positions
                    )
                    
                    if head_range == device.head_range:
                        device.kv_cache[layer_idx] = new_kv
                
                # 存储输出
                num_heads_in_range = end_head - start_head
                head_dim = config.head_dim
                attn_out_reshaped = attn_out.view(1, -1, num_heads_in_range, head_dim)
                
                for i, head_idx in enumerate(range(start_head, end_head)):
                    all_head_outputs[head_idx] = attn_out_reshaped[:, :, i, :]
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  耗时: {elapsed:.4f}s")
        
        # 简化：直接生成假的下一个 token
        next_token = 100 + step
        tokens.append(next_token)
        
        print(f"✅ 生成 token: {next_token}")
    
    print("\n" + "=" * 60)
    print("✅ Demo 完成！")
    print("=" * 60)
    print(f"\n总结:")
    print(f"  - 生成了 {num_tokens_to_generate} 个 tokens")
    print(f"  - 在第 {OFFLINE_AT_TOKEN + 1} 个 token 时设备 {OFFLINE_DEVICE} 下线")
    print(f"  - 成功复用了其他设备的 KV-Cache")
    print(f"  - 仅对丢失部分进行了重新计算")


if __name__ == "__main__":
    main()