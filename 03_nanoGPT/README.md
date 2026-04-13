# 03. nanoGPT 재현

> Karpathy, A. (2022). nanoGPT
> 원본 코드: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
> 기반 논문: GPT-2 (Radford et al., 2019)

---

## 개요

GPT-2 논문 구현(`02_GPT2`)에서 한 단계 더 나아가, **실제 텍스트 데이터로 학습하고 생성**하는 전체 파이프라인을 재현한 구현입니다.

---

## 구현 구조

```
implementation/
├── data.py       # 데이터 준비 (Shakespeare 다운로드 + character-level 토크나이징)
├── model.py      # GPT 모델 (02_GPT2 gpt2.py 기반 + configure_optimizers 추가)
├── train.py      # 학습 루프 (Cosine Warmup LR + gradient clipping + checkpoint)
└── generate.py   # 텍스트 생성 (checkpoint 로드 + argparse)
```

---

## 02_GPT2와의 차이점

| 항목 | 02_GPT2/gpt2.py | 03_nanoGPT/model.py |
|---|---|---|
| 목적 | 논문 구조 이해 | 실제 학습 가능 |
| vocab_size | 50,257 고정 | 데이터에서 자동 설정 |
| 토크나이징 | BPE | Character-level |
| 옵티마이저 | 기본 AdamW | Weight decay 파라미터 분리 |
| 데이터 파이프라인 | 없음 | data.py (bin 파일 저장) |
| 학습 루프 | 없음 | train.py (eval + checkpoint) |
| 텍스트 생성 | forward() 내 | generate.py (독립 스크립트) |

---

## 실행 방법

```bash
# 1. 데이터 준비
python data.py

# 2. 학습
python train.py

# 3. 텍스트 생성
python generate.py --prompt "ROMEO:" --max_tokens 300 --temperature 0.8
```

> **참고**: `input.txt`를 직접 같은 폴더에 넣으면 자동으로 사용됩니다.
> 다운로드: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

---

## 핵심 구현 포인트

### 1. Weight Decay 파라미터 분리

```python
# 2D 파라미터(weight)만 decay 적용, 1D(bias, LayerNorm)는 제외
decay    = {Linear.weight, Embedding.weight}
no_decay = {bias, LayerNorm.weight, LayerNorm.bias}
```

### 2. Cosine Warmup LR 스케줄

```
0 ~ warmup_iters  : 선형 증가  (0 → max_lr)
warmup ~ max_iters: Cosine 감소 (max_lr → min_lr)
```

### 3. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

---

## 권장 학습 설정

| 환경 | BLOCK_SIZE | N_EMBD | N_LAYER | N_HEAD | BATCH_SIZE |
|---|---|---|---|---|---|
| GPU | 256 | 384 | 6 | 6 | 64 |
| CPU | 128 | 128 | 4 | 4 | 32 |

---

## 참고 자료

- [nanoGPT 원본 코드](https://github.com/karpathy/nanoGPT)
- [Karpathy의 "Let's build GPT" 강의](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [GPT-2 논문](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
