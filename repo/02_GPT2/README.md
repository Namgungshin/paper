# 02. Language Models are Unsupervised Multitask Learners (GPT-2)

> Radford et al., OpenAI, 2019  
> 논문 링크: [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## 핵심 주장

> **"별도의 지도학습 없이, 충분히 큰 언어 모델은 다양한 NLP 태스크를 스스로 학습한다."**

기존 NLP 시스템이 태스크별 지도학습에 의존한 것과 달리, GPT-2는 대규모 웹 텍스트(WebText)로 사전학습만 해도 번역·요약·QA 등 다양한 태스크를 **제로샷(Zero-shot)**으로 수행할 수 있음을 보임.

---

## Attention Is All You Need와의 핵심 차이점

| 항목 | Transformer (2017) | GPT-2 (2019) |
|---|---|---|
| 구조 | Encoder + Decoder | **Decoder-only** |
| LayerNorm 위치 | Post-LN | **Pre-LN** |
| Positional Encoding | Sinusoidal (고정) | **학습 가능 Embedding** |
| Attention Mask | Padding mask | **Causal mask** |
| Activation | ReLU | **GELU** |
| Weight Tying | 없음 | **lm_head ↔ wte 가중치 공유** |
| 학습 목표 | 지도학습 (번역) | **비지도학습 (다음 토큰 예측)** |

---

## 모델 크기 (논문 Table 1)

| 이름 | 파라미터 | n_layer | n_embd | n_head |
|---|---|---|---|---|
| Small | 117M | 12 | 768 | 12 |
| Medium | 345M | 24 | 1024 | 16 |
| Large | 762M | 36 | 1280 | 20 |
| XL | 1.5B | 48 | 1600 | 25 |

---

## 구현 구조

```
implementation/
└── gpt2.py
    ├── GPT2Config          # 모델 하이퍼파라미터 (dataclass)
    ├── CausalSelfAttention # Masked Multi-Head Self-Attention
    ├── MLP                 # Position-wise FFN (GELU 활성화)
    ├── Block               # Pre-LN Transformer Block
    ├── GPT2                # Full Model (forward + generate)
    └── CosineWarmupScheduler  # LR 스케줄러
```

### 주요 구현 포인트

**1. Pre-Layer Normalization (Section 2.1)**
```python
# 원본 Transformer (Post-LN)
x = LayerNorm(x + Sublayer(x))

# GPT-2 (Pre-LN) — 학습 안정성 향상
x = x + Sublayer(LayerNorm(x))
```

**2. Causal Mask — Autoregressive 생성**
```python
# 미래 토큰을 보지 못하도록 상삼각 마스킹
mask = torch.tril(torch.ones(T, T))  # (T, T)
attn = attn.masked_fill(mask == 0, float('-inf'))
```

**3. Weight Tying — 임베딩 가중치 공유**
```python
# LM Head와 Token Embedding 가중치 공유 → 파라미터 절약
self.lm_head.weight = self.transformer['wte'].weight
```

**4. Cosine Warmup LR 스케줄러 (Section 2.3)**
```python
# warmup: 선형 증가 → cosine decay
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * ratio))
```

---

## 실행 방법

```bash
# 의존성 설치
pip install torch

# Smoke Test 실행
python implementation/gpt2.py
```

**예상 출력:**
```
GPT-2 파라미터 수: 0.9M  (tiny 테스트 설정)
Input shape  : torch.Size([2, 64])
Logits shape : torch.Size([2, 64, 1000])
Loss         : 6.9410
이론값(random): 6.9078  ← 거의 일치 ✅
1 step 학습 완료 ✅
```

> **참고**: Smoke Test는 메모리 절약을 위해 tiny 설정(vocab=1000, n_embd=128, n_layer=4)을 사용합니다.  
> 실제 GPT-2 Small(117M) 설정은 코드 내 `GPT2Config` 주석 참고.

---

## 학습 흐름 요약

```
WebText (40GB)
    ↓
Token Embedding + Position Embedding
    ↓
N × [Pre-LN → Masked Self-Attention → Pre-LN → FFN]
    ↓
Final LayerNorm
    ↓
LM Head (vocab_size) — wte와 가중치 공유
    ↓
Cross-Entropy Loss (다음 토큰 예측)
```

---

## 참고 자료

- [원본 논문 PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenAI 공식 코드](https://github.com/openai/gpt-2)
- [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT) — 이 구현의 구조적 참고
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
