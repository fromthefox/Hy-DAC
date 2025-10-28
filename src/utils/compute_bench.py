import torch
import time
import torch.nn as nn

# 模拟Llama单层：简化Attention + FFN，hidden=4096, fp16
class SimpleLlamaLayer(nn.Module):
    def __init__(self, hidden=4096, num_heads=32):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4*hidden),
            nn.GELU(),
            nn.Linear(4*hidden, hidden)
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
    
    def forward(self, x):
        # x: (batch=1, seq=1, hidden)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

if __name__ == "__main__":
    device = torch.device("cpu")
    dtype = torch.float16  # fp16
    model = SimpleLlamaLayer().to(device=device, dtype=dtype)
    x = torch.randn(1, 1, 4096, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    # 测量100次
    times = []
    for _ in range(100):
        start = time.time()
        _ = model(x)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    C = 1.0 / avg_time  # 层/秒
    print(f"Avg layer time: {avg_time:.6f}s, C: {C:.6f}")