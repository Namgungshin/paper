# Paper Implementation Portfolio

딥러닝 핵심 논문을 읽고, 한국어로 요약하고, PyTorch로 직접 구현한 포트폴리오입니다.

---

## 논문 목록

| # | 논문 | 연도 | 핵심 기여 | 구현 |
|---|---|---|---|---|
| 01 | [Attention Is All You Need](./01_Attention_Is_All_You_Need/) | 2017 | Transformer 아키텍처 제안 | ✅ |
| 02 | [Language Models are Unsupervised Multitask Learners (GPT-2)](./02_GPT2/) | 2019 | Decoder-only LM, Zero-shot 멀티태스크 | ✅ |
| 03 | [nanoGPT 재현](./03_nanoGPT/) | 2022 | GPT-2 실제 학습 파이프라인 전체 재현 | ✅ |

---

## 학습 로드맵

```
Attention Is All You Need (Transformer 원리)
        ↓
GPT-2 (Decoder-only, Pre-LN, Zero-shot)
        ↓
nanoGPT 재현 (실제 학습 파이프라인)
        ↓
BERT (Encoder 관점 보완) (예정)
        ↓
LoRA / RAG (예정)
```

---

## 구조

```
.
├── 01_Attention_Is_All_You_Need/
│   ├── README.md
│   └── implementation/transformer.py
├── 02_GPT2/
│   ├── README.md
│   └── implementation/gpt2.py
├── 03_nanoGPT/
│   ├── README.md
│   └── implementation/
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       └── generate.py
└── README.md
```

---

## 실행 환경

```bash
pip install torch numpy
```
