import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
import einx
import einops
import math

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        sigma = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weights, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot(
            "d_out d_in, ... d_in -> ... d_out",
            self.weights, x
        )

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.weights = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        torch.nn.init.trunc_normal_(self.weights, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x * x
        rms = einx.mean("... [i]", rms , keepdims=True)
        rms = rms + self.eps
        rms = torch.rsqrt(rms)
        res = x * rms * self.weights
        return res.to(in_dtype)
    
def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    """
        SiLU(x) = x * sigmoid(x)
        SwiGLU = W2(SiLU(W1x) element-wise-mul W3x) 
        W1, W3 = [d_ff, d_model]
        W2 = [d_model, d_ff]
        d_ff = 8/3 * d_model
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device)
        self.w2 = Linear(d_ff, d_model, device=device)
        self.w3 = Linear(d_model, d_ff, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1.forward(x)
        b = self.w3.forward(x)
        return self.w2(silu(a) * b)
    
class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        factory_kwargs = {"device": device}
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.rope_buffer = torch.zeros((max_seq_len, d_k // 2, 2, 2))
        # print(d_k, max_seq_len)
        for i in range(max_seq_len):
            for k in range(1, d_k // 2 + 1):
                theta_ik = i * math.pow(self.theta, -(2*k-2)/self.d_k)
                # theta_ik = i / (self.theta ** (2*(k-1)/self.d_k))
                self.rope_buffer[i,k - 1] = torch.Tensor(
                    [[math.cos(theta_ik), -math.sin(theta_ik)],
                    [math.sin(theta_ik), math.cos(theta_ik)]]
                )
        self.rope_buffer = self.rope_buffer.to(device=device)

        # for _forward (just for debugging)
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        freq = torch.pow(self.theta, -torch.arange(0, d_k, 2).float() / d_k)
        angles = pos[:, None] * freq[None, :]  # [max_seq_len, dim/2]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
    
    def _forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        
        cos = self.cos[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k/2)
        # Step 2: 拆分偶/奇通道
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # Step 3: 应用旋转变换
        x_out = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        # Step 4: 合并最后两维
        return x_out.flatten(-2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # return self._forward(x,token_positions)
        # ref = self._forward(x, token_positions)
        rope = einx.get_at("[total] half_d_k a b, ... seq -> ... seq half_d_k a b", self.rope_buffer, token_positions)
        x = einx.rearrange("... seq (half_d_k n) -> ... seq half_d_k n 1", x, n=2)
        x = rope.matmul(x) # matmul会自动广播前缀，使用einx.dot需要手动修改使得前缀匹配
        x = einx.rearrange("... a b c -> ... (a b c)", x)
        return x

def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    in_dtype = x.dtype
    x = x.to(torch.float64)
    max_x = torch.max(x, dim, keepdim=True).values
    x = x - max_x
    x = torch.exp(x)
    sum = torch.sum(x, dim, keepdim=True)
    return (x / sum).to(in_dtype)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    QK = Q.matmul(K.transpose(-1, -2)) / math.sqrt(d_k)
    QK = QK.masked_fill(mask == False, -torch.inf)
    QK = softmax(QK, -1)
    return QK.matmul(V)

class CausalMHA(torch.nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None,
        device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.d_k = self.d_v = self.d_model // self.num_heads
        self.q_proj = Linear(self.d_model, self.d_k * self.num_heads, **factory_kwargs)
        self.k_proj = Linear(self.d_model, self.d_k * self.num_heads, **factory_kwargs)
        self.v_proj = Linear(self.d_model, self.d_v * self.num_heads, **factory_kwargs)
        self.o_proj = Linear(self.d_v * self.num_heads, self.d_model, **factory_kwargs)

    def forward(self, 
        x: Float[Tensor, "... seq d_model"],
        token_positions: Int[Tensor, "... seq"] | None = None
    ) -> Float[Tensor, "... seq d_model"]:
        seq = x.shape[-2]
        Q = self.q_proj.forward(x)
        Q = einx.rearrange("... seq (n head) -> ... n seq head", Q, n=self.num_heads)
        K = self.k_proj.forward(x)
        K = einx.rearrange("... seq (n head) -> ... n seq head", K, n=self.num_heads)
        V = self.v_proj.forward(x)
        V = einx.rearrange("... seq (n head) -> ... n seq head", V, n=self.num_heads)
        if token_positions is not None:
            assert(self.rope)
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        attn = scaled_dot_product_attention(Q, K, V, mask)
        attn = einx.rearrange("... n seq head -> ... seq (n head)", attn, n=self.num_heads)
        return self.o_proj.forward(attn)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int,
        d_ff: int, rope: RoPE | None = None, device = None, dtype = None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope

        self.ln1 = RMSNorm(self.d_model, **factory_kwargs)
        self.attn = CausalMHA(self.d_model, self.num_heads, self.rope, **factory_kwargs)
        self.ln2 = RMSNorm(self.d_model, **factory_kwargs)
        self.ffn = SwiGLU(self.d_model, self.d_ff, **factory_kwargs)
        
    def forward(self, 
        x: Float[Tensor, " batch sequence_length d_model"]
     ) -> Float[Tensor, " batch sequence_length d_model"]:
        seq =  x.shape[-2]
        y = x + self.attn(self.ln1(x), torch.arange(0, seq, device=x.device, dtype=torch.int))
        res = y + self.ffn(self.ln2(y))
        return res
    
class Transformer(torch.nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None, dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)
        self.rope = RoPE(rope_theta, d_model // num_heads, context_length, device=device)
        self.layers = [
            TransformerBlock(d_model, num_heads, d_ff, self.rope, **factory_kwargs)
            for _ in range(num_layers)
        ]
        self.ln_final = RMSNorm(d_model, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(self, 
        x: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x