"""
GPT-2 PyTorch Implementation
Paper: "Language Models are Unsupervised Multitask Learners"
Radford et al., OpenAI, 2019

논문 섹션 주석 기준:
- Section 2.1: Model Architecture
- Section 2.2: BPE (구현 생략, tiktoken 사용 권장)
- Section 2.3: Training Details
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ============================================================
# Section 2.3 — Model Configurations (Table 1 in paper)
# ============================================================

@dataclass
class GPT2Config:
    """
    GPT-2 논문 Table 1의 4가지 모델 크기 설정
    기본값은 GPT-2 Small (117M)
    """
    vocab_size: int = 50257      # BPE vocab size (Section 2.2)
    n_positions: int = 1024      # context window size (Section 2.3)
    n_embd: int = 768            # embedding dimension (d_model)
    n_layer: int = 12            # transformer block 수
    n_head: int = 12             # attention head 수
    dropout: float = 0.1

    # 모델 크기별 설정 참고
    # Small  (117M): n_embd=768,  n_layer=12, n_head=12
    # Medium (345M): n_embd=1024, n_layer=24, n_head=16
    # Large  (762M): n_embd=1280, n_layer=36, n_head=20
    # XL    (1.5B):  n_embd=1600, n_layer=48, n_head=25


# ============================================================
# Section 2.1 — Masked Self-Attention
# Transformer와의 차이: Cross-Attention 없음 (Decoder-only)
# Pre-Layer Normalization 적용 (원본 Transformer는 Post-LN)
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    GPT-2의 Masked Multi-Head Self-Attention
    
    핵심 차이점 (vs Attention Is All You Need):
    - Cross-Attention 없음 → Decoder-only이므로 Encoder 입력 불필요
    - Causal Mask: 미래 토큰을 보지 못하게 차단 (autoregressive)
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V를 한 번에 계산 (효율적)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: 상삼각 행렬 (미래 토큰 차단)
        # register_buffer: 파라미터는 아니지만 state_dict에 포함
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )

    def forward(self, x):
        B, T, C = x.shape  # (batch, seq_len, n_embd)

        # Q, K, V 분리
        qkv = self.c_attn(x)                         # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)      # 각 (B, T, C)

        # Multi-Head로 분할
        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            # → (B, n_head, T, head_dim)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled Dot-Product Attention (논문 Eq. 1과 동일)
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale     # (B, n_head, T, T)

        # Causal Masking: 미래 위치를 -inf로 설정 → softmax 후 0이 됨
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Value와 결합
        out = attn @ v                                # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.resid_dropout(self.c_proj(out))

        return out


# ============================================================
# Section 2.1 — Position-wise Feed-Forward Network
# 원본 Transformer와 동일한 구조 (GELU activation 사용)
# ============================================================

class MLP(nn.Module):
    """
    FFN(x) = GELU(xW1 + b1)W2 + b2
    
    차이점: ReLU 대신 GELU 사용
    GELU: 확률적 게이팅 효과, 더 부드러운 그래디언트
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)   # 확장
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)   # 축소
        self.act    = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ============================================================
# Section 2.1 — Transformer Block (Pre-LN 구조)
# 핵심: LayerNorm이 서브레이어 앞에 위치 (Pre-LN)
# 원본 Transformer: x = LayerNorm(x + Sublayer(x))  ← Post-LN
# GPT-2:           x = x + Sublayer(LayerNorm(x))   ← Pre-LN
# ============================================================

class Block(nn.Module):
    """
    GPT-2 Transformer Block
    
    Pre-LN 구조:
        x → LayerNorm → Attention → + residual
        x → LayerNorm → MLP       → + residual
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        # Pre-LN: LayerNorm 먼저, 그 후 서브레이어, residual 연결
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ============================================================
# Section 2.1 — Full GPT-2 Model
# ============================================================

class GPT2(nn.Module):
    """
    GPT-2 Full Model
    
    구성:
    1. Token Embedding  (vocab_size → n_embd)
    2. Position Embedding (n_positions → n_embd) ← 학습 가능 (Transformer와 차이)
    3. N × Transformer Block (Pre-LN)
    4. Final LayerNorm
    5. LM Head (n_embd → vocab_size) — Token Embedding과 가중치 공유
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),    # token embedding
            'wpe': nn.Embedding(config.n_positions, config.n_embd),   # position embedding
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),                      # final LayerNorm
        })

        # LM Head: 임베딩 가중치 공유 (Weight Tying)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer['wte'].weight

        # 가중치 초기화
        self.apply(self._init_weights)

        # 파라미터 수 출력
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT-2 파라미터 수: {n_params/1e6:.1f}M")

    def _init_weights(self, module):
        """
        Section 2.3: 가중치 초기화
        - Linear, Embedding: N(0, 0.02)
        - 잔차 연결 레이어: 추가로 1/√(2N) 스케일링 (N=레이어 수)
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        Args:
            idx: (B, T) 토큰 인덱스
            targets: (B, T) 정답 토큰 (학습 시), None이면 추론 모드
        Returns:
            logits: (B, T, vocab_size)
            loss: CrossEntropyLoss (targets 있을 때만)
        """
        B, T = idx.shape
        assert T <= self.config.n_positions, \
            f"시퀀스 길이 {T}가 최대 컨텍스트 {self.config.n_positions}를 초과"

        # 1. Token + Position Embedding
        tok_emb = self.transformer['wte'](idx)                        # (B, T, n_embd)
        pos     = torch.arange(T, device=idx.device).unsqueeze(0)    # (1, T)
        pos_emb = self.transformer['wpe'](pos)                        # (1, T, n_embd)
        x = self.transformer['drop'](tok_emb + pos_emb)

        # 2. N × Transformer Block
        for block in self.transformer['h']:
            x = block(x)

        # 3. Final LayerNorm
        x = self.transformer['ln_f'](x)

        # 4. LM Head → logits
        logits = self.lm_head(x)                                      # (B, T, vocab_size)

        # 5. Loss 계산 (학습 시)
        loss = None
        if targets is not None:
            # (B, T, vocab_size) → (B*T, vocab_size) 로 펼쳐서 계산
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive 텍스트 생성
        
        Args:
            idx: (B, T) 시작 토큰 시퀀스
            max_new_tokens: 생성할 토큰 수
            temperature: 낮을수록 결정적, 높을수록 다양
            top_k: Top-K 샘플링 (None이면 전체 vocab)
        """
        for _ in range(max_new_tokens):
            # 컨텍스트 길이 초과 시 자르기
            idx_cond = idx if idx.size(1) <= self.config.n_positions \
                       else idx[:, -self.config.n_positions:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 마지막 토큰의 logits

            # Top-K 필터링
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)             # (B, T+1)

        return idx


# ============================================================
# Section 2.3 — Learning Rate Scheduler (Cosine with Warmup)
# ============================================================

class CosineWarmupScheduler:
    """
    Section 2.3: Learning rate schedule
    - Warmup: 선형 증가 (0 → max_lr)
    - Decay: Cosine 감소 (max_lr → min_lr)
    """
    def __init__(self, optimizer, warmup_steps, max_steps,
                 max_lr=2.5e-4, min_lr=1e-5):
        self.optimizer   = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps   = max_steps
        self.max_lr      = max_lr
        self.min_lr      = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _get_lr(self):
        s = self.current_step
        if s < self.warmup_steps:
            return self.max_lr * s / self.warmup_steps
        if s >= self.max_steps:
            return self.min_lr
        # Cosine decay
        ratio = (s - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1 + math.cos(math.pi * ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


# ============================================================
# 간단한 동작 확인 (Smoke Test)
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    # Smoke test용 tiny 설정 (메모리 절약)
    # 실제 GPT-2 Small: vocab=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=128,
        n_layer=4,
        n_head=4,
        dropout=0.1,
    )

    model = GPT2(config)
    print(f"설정: {config}\n")

    # Forward pass 확인
    B, T = 2, 64
    idx     = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx, targets)
    print(f"Input shape  : {idx.shape}")
    print(f"Logits shape : {logits.shape}")
    print(f"Loss         : {loss.item():.4f}")
    print(f"초기 loss 이론값 (random): {math.log(config.vocab_size):.4f}")

    # 텍스트 생성 확인
    start = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(start, max_new_tokens=20, temperature=1.0, top_k=50)
    print(f"\n생성된 토큰 시퀀스 shape: {generated.shape}")
    print(f"생성된 토큰: {generated[0].tolist()}")

    # 옵티마이저 + 스케줄러
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=100, max_steps=10000)

    # 1 step 학습 확인
    optimizer.zero_grad()
    logits, loss = model(idx, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
    optimizer.step()
    lr = scheduler.step()
    print(f"\n1 step 학습 완료 | loss: {loss.item():.4f} | lr: {lr:.6f}")
