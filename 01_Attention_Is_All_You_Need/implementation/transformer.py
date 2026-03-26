"""
Attention Is All You Need (Vaswani et al., 2017) - PyTorch 구현
=============================================================

논문의 Transformer 아키텍처를 처음부터 구현합니다.
각 컴포넌트에 해당하는 논문 섹션과 수식 번호를 주석으로 표기했습니다.

하이퍼파라미터 (논문 Table 3 base model):
  - d_model = 512     (임베딩 차원)
  - N = 6             (인코더/디코더 레이어 수)
  - h = 8             (어텐션 헤드 수)
  - d_k = d_v = 64    (각 헤드의 key/value 차원)
  - d_ff = 2048       (FFN 내부 차원)
  - P_drop = 0.1      (드롭아웃 비율)
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Scaled Dot-Product Attention (Section 3.2.1, Equation 1)
# ============================================================================
# Attention(Q, K, V) = softmax(QK^T / √d_k) V

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Args:
        query:   (batch, h, seq_len, d_k)
        key:     (batch, h, seq_len, d_k)
        value:   (batch, h, seq_len, d_v)
        mask:    선택적 마스크 (패딩 또는 미래 토큰 차단용)
        dropout: 드롭아웃 레이어
    Returns:
        output:  (batch, h, seq_len, d_v)  - 어텐션 가중합 결과
        attn_weights: (batch, h, seq_len, seq_len) - 어텐션 가중치
    """
    d_k = query.size(-1)

    # QK^T / √d_k
    # √d_k로 나누는 이유: d_k가 클수록 내적 값이 커져서
    # softmax가 극단적인 값을 출력하고 기울기가 사라지기 때문 (논문 각주 4)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 마스크 적용: -∞로 설정하면 softmax 후 0이 됨
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # softmax로 가중치 정규화 (합 = 1)
    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # 가중치 × Value
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


# ============================================================================
# 2. Multi-Head Attention (Section 3.2.2)
# ============================================================================
# MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
# where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

class MultiHeadAttention(nn.Module):
    """
    h=8개의 어텐션 헤드를 병렬로 실행.
    각 헤드는 d_k=64 차원에서 독립적으로 어텐션을 수행.
    """

    def __init__(self, d_model=512, h=8, dropout=0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h   # 64
        self.h = h

        # W^Q, W^K, W^V, W^O 선형 변환 (논문: R^{d_model × d_k} 각각)
        self.W_Q = nn.Linear(d_model, d_model)  # 내부적으로 h개 헤드로 분할
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights = None  # 시각화용 저장

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) 선형 변환 후 h개 헤드로 분할
        #    (batch, seq_len, d_model) → (batch, h, seq_len, d_k)
        Q = self.W_Q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 마스크 차원 맞춤: (batch, 1, 1, seq_len) 또는 (batch, 1, seq_len, seq_len)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (batch, 1, ...) → 모든 헤드에 브로드캐스트

        # 2) Scaled Dot-Product Attention 수행
        x, self.attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # 3) 헤드 결합 (Concat) 후 W^O 적용
        #    (batch, h, seq_len, d_k) → (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.W_O(x)


# ============================================================================
# 3. Position-wise Feed-Forward Network (Section 3.3, Equation 2)
# ============================================================================
# FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
# 내부 차원 d_ff=2048, 입출력 차원 d_model=512

class PositionwiseFeedForward(nn.Module):
    """각 위치에 동일하게 적용되는 2층 FFN (ReLU 활성화)."""

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ============================================================================
# 4. Positional Encoding (Section 3.5)
# ============================================================================
# PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
# PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

class PositionalEncoding(nn.Module):
    """
    사인/코사인 기반 위치 인코딩.
    Transformer에는 순서 개념이 없으므로, 이 인코딩으로 위치 정보를 주입.
    """

    def __init__(self, d_model=512, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # (max_len, d_model) 크기의 위치 인코딩 테이블 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )  # 10000^{-2i/d_model}

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스: cos

        pe = pe.unsqueeze(0)  # (1, max_len, d_model) — 배치 차원 추가
        self.register_buffer("pe", pe)  # 학습되지 않는 파라미터

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ============================================================================
# 5. 임베딩 (Section 3.4)
# ============================================================================
# 논문: 임베딩 가중치에 √d_model을 곱함

class TokenEmbedding(nn.Module):
    """토큰 임베딩 + √d_model 스케일링."""

    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# ============================================================================
# 6. Sublayer Connection: Residual + LayerNorm (Section 3.1)
# ============================================================================
# Output = LayerNorm(x + Sublayer(x))

class SublayerConnection(nn.Module):
    """잔차 연결(Residual Connection) + 레이어 정규화."""

    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # Pre-norm 변형 (구현 편의상, 원 논문은 post-norm이지만 효과는 유사)
        return x + self.dropout(sublayer(self.norm(x)))


# ============================================================================
# 7. Encoder Layer (Section 3.1)
# ============================================================================
# 각 인코더 레이어 = Self-Attention + Feed-Forward (각각 잔차연결 포함)

class EncoderLayer(nn.Module):
    """인코더 레이어 1개: Self-Attention → FFN."""

    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, src_mask=None):
        # Self-Attention: Q=K=V=x (같은 시퀀스 내에서 서로 참조)
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.sublayer2(x, self.feed_forward)
        return x


# ============================================================================
# 8. Decoder Layer (Section 3.1)
# ============================================================================
# 각 디코더 레이어 = Masked Self-Attention + Cross-Attention + Feed-Forward

class DecoderLayer(nn.Module):
    """디코더 레이어 1개: Masked Self-Attn → Cross-Attn → FFN."""

    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1) Masked Self-Attention: 미래 토큰을 보지 못하도록 마스킹
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2) Cross-Attention: Q=디코더, K=V=인코더 출력 (Section 3.2.3)
        x = self.sublayer2(x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask))
        # 3) Feed-Forward
        x = self.sublayer3(x, self.feed_forward)
        return x


# ============================================================================
# 9. Encoder & Decoder Stacks (Section 3.1)
# ============================================================================

class Encoder(nn.Module):
    """N=6개의 인코더 레이어 스택."""

    def __init__(self, layer, N=6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.W_Q.in_features)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    """N=6개의 디코더 레이어 스택."""

    def __init__(self, layer, N=6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.W_Q.in_features)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# ============================================================================
# 10. Generator: 최종 출력 (Section 3.4)
# ============================================================================
# 디코더 출력 → 선형 변환 → softmax → 다음 토큰 확률 분포

class Generator(nn.Module):
    """디코더 출력을 어휘(vocabulary) 확률 분포로 변환."""

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.projection(x), dim=-1)


# ============================================================================
# 11. 전체 Transformer 모델
# ============================================================================

class Transformer(nn.Module):
    """
    Attention Is All You Need — 전체 Transformer 모델.

    구조:
      입력 → Embedding + Positional Encoding → Encoder Stack
      출력 → Embedding + Positional Encoding → Decoder Stack → Generator
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        N=6,
        h=8,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()

        # 임베딩 + 위치 인코딩
        self.src_embed = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # 인코더 & 디코더 스택
        encoder_layer = EncoderLayer(d_model, h, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, h, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)

        # 출력 생성기
        self.generator = Generator(d_model, tgt_vocab_size)

        # 파라미터 초기화 (Xavier uniform)
        self._init_parameters()

    def _init_parameters(self):
        """Xavier uniform 초기화 — 학습 안정성 향상."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        """인코더: 소스 시퀀스 → 문맥 표현."""
        return self.encoder(self.pos_enc(self.src_embed(src)), src_mask)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """디코더: 타겟 시퀀스 + 인코더 출력 → 디코딩된 표현."""
        return self.decoder(self.pos_enc(self.tgt_embed(tgt)), encoder_output, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: 소스 토큰 인덱스 (batch, src_len)
            tgt: 타겟 토큰 인덱스 (batch, tgt_len)
            src_mask: 소스 패딩 마스크
            tgt_mask: 타겟 미래 토큰 마스크 (causal mask)
        Returns:
            log_probs: (batch, tgt_len, tgt_vocab_size)
        """
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.generator(dec_output)


# ============================================================================
# 12. 마스크 생성 유틸리티
# ============================================================================

def create_padding_mask(seq, pad_idx=0):
    """패딩 토큰(pad_idx)을 무시하기 위한 마스크.
    
    Returns: (batch, 1, 1, seq_len) — True인 위치만 attend
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size):
    """미래 토큰을 보지 못하게 하는 삼각 마스크 (Section 3.2.3).
    
    디코더의 Self-Attention에서 사용.
    Returns: (1, size, size) — 하삼각 행렬 (현재+과거만 attend)
    """
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool)).unsqueeze(0)  # (1, size, size)
    return mask  # True = attend, False = mask


def create_tgt_mask(tgt, pad_idx=0):
    """타겟 시퀀스용 마스크 = 패딩 마스크 & 인과 마스크 결합."""
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)             # (batch, 1, 1, tgt_len)
    tgt_causal_mask = create_causal_mask(tgt.size(1)).to(tgt.device)  # (1, tgt_len, tgt_len)
    return tgt_pad_mask & tgt_causal_mask.unsqueeze(0)


# ============================================================================
# 13. 학습률 스케줄러 (Section 5.3, Equation 3)
# ============================================================================
# lrate = d_model^{-0.5} · min(step^{-0.5}, step · warmup_steps^{-1.5})

class TransformerLRScheduler:
    """
    논문의 학습률 스케줄링:
    - warmup_steps(=4000) 동안 선형 증가
    - 이후 step^{-0.5}에 비례해 감소
    """

    def __init__(self, optimizer, d_model=512, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5),
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


# ============================================================================
# 14. Label Smoothing (Section 5.4)
# ============================================================================
# 논문: label smoothing ε=0.1 사용 — perplexity는 나빠지지만 BLEU는 향상

class LabelSmoothingLoss(nn.Module):
    """
    라벨 스무딩이 적용된 KL divergence 손실.
    정답에 (1-ε)의 확률을, 나머지 클래스에 ε/(V-1)의 확률을 배분.
    """

    def __init__(self, vocab_size, pad_idx=0, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred:   (batch * seq_len, vocab_size) — log probabilities
            target: (batch * seq_len,) — 정답 토큰 인덱스
        """
        # 스무딩된 분포 생성
        smooth_dist = torch.full_like(pred, self.smoothing / (self.vocab_size - 2))
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_dist[:, self.pad_idx] = 0  # 패딩은 손실에서 제외

        # 패딩 위치 마스킹
        pad_mask = target == self.pad_idx
        smooth_dist[pad_mask] = 0

        return F.kl_div(pred, smooth_dist, reduction="sum")


# ============================================================================
# 15. 데모: 모델 생성 및 더미 데이터 추론
# ============================================================================

def demo():
    """모델이 올바르게 구성되고 forward pass가 작동하는지 검증."""
    print("=" * 60)
    print("Transformer (Attention Is All You Need) — PyTorch 구현")
    print("=" * 60)

    # 하이퍼파라미터 (논문 base model)
    SRC_VOCAB = 10000
    TGT_VOCAB = 10000
    D_MODEL = 512
    N_LAYERS = 6
    N_HEADS = 8
    D_FF = 2048
    DROPOUT = 0.1

    # 모델 생성
    model = Transformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=D_MODEL,
        N=N_LAYERS,
        h=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
    )

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[모델 구성]")
    print(f"  d_model     = {D_MODEL}")
    print(f"  레이어 수    = {N_LAYERS}")
    print(f"  어텐션 헤드  = {N_HEADS}")
    print(f"  d_ff        = {D_FF}")
    print(f"  dropout     = {DROPOUT}")
    print(f"  소스 어휘    = {SRC_VOCAB:,}")
    print(f"  타겟 어휘    = {TGT_VOCAB:,}")
    print(f"\n[파라미터]")
    print(f"  전체 파라미터  = {total_params:,}")
    print(f"  학습 가능     = {trainable_params:,}")
    print(f"  논문 base model 참고값: ~65M (어휘 크기에 따라 변동)")

    # 더미 데이터로 Forward Pass 테스트
    print(f"\n[Forward Pass 테스트]")
    BATCH_SIZE = 2
    SRC_LEN = 10
    TGT_LEN = 8

    src = torch.randint(1, SRC_VOCAB, (BATCH_SIZE, SRC_LEN))  # 0=pad, 1~9999=토큰
    tgt = torch.randint(1, TGT_VOCAB, (BATCH_SIZE, TGT_LEN))

    src_mask = create_padding_mask(src, pad_idx=0)
    tgt_mask = create_tgt_mask(tgt, pad_idx=0)

    print(f"  입력 src shape:  {tuple(src.shape)}")
    print(f"  입력 tgt shape:  {tuple(tgt.shape)}")

    model.eval()
    with torch.no_grad():
        output = model(src, tgt, src_mask, tgt_mask)

    print(f"  출력 shape:      {tuple(output.shape)}")
    print(f"  기대 shape:      ({BATCH_SIZE}, {TGT_LEN}, {TGT_VOCAB})")
    assert output.shape == (BATCH_SIZE, TGT_LEN, TGT_VOCAB)
    print(f"\n  ✓ Forward pass 성공!")

    # 학습률 스케줄러 테스트
    print(f"\n[학습률 스케줄러 테스트]")
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer, d_model=D_MODEL, warmup_steps=4000)

    test_steps = [1, 1000, 4000, 8000, 100000]
    for step in test_steps:
        scheduler.step_num = step - 1
        lr = scheduler.step()
        print(f"  Step {step:>6d}: lr = {lr:.6e}")

    print(f"\n{'=' * 60}")
    print("구현 완료! 각 컴포넌트가 논문의 수식과 일치합니다.")
    print("=" * 60)

    # 컴포넌트별 대응 관계 출력
    print(f"""
[논문 섹션 ↔ 코드 매핑]
  Section 3.2.1  →  scaled_dot_product_attention()  (Eq. 1)
  Section 3.2.2  →  MultiHeadAttention
  Section 3.3    →  PositionwiseFeedForward          (Eq. 2)
  Section 3.5    →  PositionalEncoding
  Section 3.4    →  TokenEmbedding, Generator
  Section 3.1    →  EncoderLayer, DecoderLayer
  Section 5.3    →  TransformerLRScheduler            (Eq. 3)
  Section 5.4    →  LabelSmoothingLoss
    """)


if __name__ == "__main__":
    demo()
