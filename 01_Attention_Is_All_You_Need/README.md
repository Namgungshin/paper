# Attention Is All You Need (2017)

> Vaswani et al., NIPS 2017  
> 원문: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## 한 줄 요약

RNN/CNN 없이 **Attention 메커니즘만으로** 시퀀스를 처리하는 Transformer 아키텍처를 제안 → 병렬화 가능 + 장거리 의존성 학습 용이 → 더 빠르고 더 좋은 성능

---

## 1. 문제 의식 (왜 만들었나?)

기존 시퀀스 처리의 지배적 모델인 **RNN (LSTM, GRU)**에는 두 가지 핵심 한계가 있었다:

- **순차 처리 병목**: 이전 시점의 계산이 끝나야 다음 시점을 처리할 수 있어 GPU 병렬화가 불가능. 긴 시퀀스일수록 학습 시간이 크게 증가.
- **장거리 의존성 학습 어려움**: 문장 내 멀리 떨어진 단어 간의 관계를 학습하려면 신호가 여러 단계를 거쳐야 하므로 정보가 희석됨.

**CNN 기반 모델** (ByteNet, ConvS2S)은 병렬화는 가능하지만, 임의의 두 위치를 연결하려면 O(log n) 이상의 레이어가 필요.

---

## 2. 핵심 아이디어

### Transformer 전체 구조

```
[입력] → Embedding + Positional Encoding → Encoder (×6) ─┐
                                                          ├→ Cross-Attention
[출력] → Embedding + Positional Encoding → Decoder (×6) ─┘→ Linear → Softmax → [예측]
```

- **인코더**: 입력 시퀀스를 문맥 표현으로 변환 (Self-Attention → FFN × 6 레이어)
- **디코더**: 인코더 출력을 참조하며 출력 시퀀스 생성 (Masked Self-Attn → Cross-Attn → FFN × 6 레이어)

### Scaled Dot-Product Attention (Eq. 1)

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- Q(Query), K(Key), V(Value) 벡터 간의 유사도를 계산하여 가중합
- `√d_k`로 나누는 이유: d_k가 클수록 내적 값이 커져 softmax 기울기가 소실되기 때문

### Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

- h=8개의 헤드가 **서로 다른 관점**에서 독립적으로 어텐션 수행
- 각 헤드: d_k = d_v = d_model / h = 64 차원
- 하나의 헤드는 문법적 관계를, 다른 헤드는 의미적 관계를 학습할 수 있음

### Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

- Transformer에는 순서 개념이 없으므로 사인/코사인 함수로 위치 정보 주입
- 상대적 위치를 선형 변환으로 표현 가능 → 학습 시 본 적 없는 길이에도 일반화 가능

### Position-wise Feed-Forward Network (Eq. 2)

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

- 각 위치에 동일하게 적용되는 2층 네트워크 (내부 차원 d_ff=2048)

---

## 3. Self-Attention vs RNN vs CNN 비교 (Table 1)

| 비교 항목 | Self-Attention | RNN | CNN |
|---|---|---|---|
| 임의의 두 위치 간 경로 길이 | O(1) | O(n) | O(log_k(n)) |
| 병렬화 가능 최소 연산 | O(1) | O(n) | O(1) |
| 레이어당 계산 복잡도 | O(n²·d) | O(n·d²) | O(k·n·d²) |

- Self-Attention의 핵심 장점: **모든 위치 쌍을 한 번에 연결** (경로 길이 O(1))
- n < d인 일반적인 경우 (word-piece/BPE 기반), Self-Attention이 RNN보다 계산 효율적

---

## 4. 학습 설정

| 항목 | 값 |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹) |
| 학습률 | warmup 4000 스텝 → 이후 step⁻⁰·⁵ 비례 감소 |
| Dropout | P_drop = 0.1 (base) / 0.3 (big) |
| Label Smoothing | ε = 0.1 |
| 하드웨어 | 8× NVIDIA P100 GPU |
| 학습 시간 | base: 12시간 (100K steps) / big: 3.5일 (300K steps) |

---

## 5. 실험 결과 (Table 2)

### 영어→독일어 (WMT 2014)

| 모델 | BLEU | 학습 비용 (FLOPs) |
|---|---|---|
| GNMT + RL Ensemble | 26.30 | 1.8 × 10²⁰ |
| ConvS2S Ensemble | 26.36 | 7.7 × 10¹⁹ |
| **Transformer (big)** | **28.4** | **2.3 × 10¹⁹** |

### 영어→프랑스어 (WMT 2014)

| 모델 | BLEU | 학습 비용 (FLOPs) |
|---|---|---|
| GNMT + RL Ensemble | 41.16 | 1.1 × 10²¹ |
| **Transformer (big)** | **41.0** | **2.3 × 10¹⁹** |

- 기존 최고 성능(앙상블 포함) 대비 **BLEU 2점 이상 향상** (EN→DE)
- 학습 비용은 기존 모델의 **1/4 이하**

---

## 6. 코드 구현

`implementation/transformer.py`에 논문의 전체 아키텍처를 PyTorch로 구현했습니다.

### 구현 컴포넌트

| 논문 섹션 | 클래스/함수 | 핵심 |
|---|---|---|
| §3.2.1 | `scaled_dot_product_attention()` | Eq. 1 |
| §3.2.2 | `MultiHeadAttention` | 8 heads, d_k=64 |
| §3.3 | `PositionwiseFeedForward` | Eq. 2 |
| §3.5 | `PositionalEncoding` | sin/cos |
| §3.1 | `EncoderLayer`, `DecoderLayer` | Residual + LayerNorm |
| §3.4 | `TokenEmbedding`, `Generator` | √d_model scaling |
| §5.3 | `TransformerLRScheduler` | Eq. 3 |
| §5.4 | `LabelSmoothingLoss` | ε=0.1 |

### 실행

```bash
pip install torch
python implementation/transformer.py
```

---

## 7. 이 논문이 중요한 이유

이 논문 이후 등장한 거의 모든 주요 언어 모델이 Transformer 아키텍처를 기반으로 한다:

- **BERT** (2018) — Transformer의 인코더만 사용
- **GPT 시리즈** (2018~) — Transformer의 디코더만 사용
- **T5, BART** — 인코더-디코더 모두 사용
- **Vision Transformer (ViT)** — 이미지에도 적용

현대 AI의 기반이 되는 가장 중요한 논문 중 하나.
