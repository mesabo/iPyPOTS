# LLM4IMP: Prompted LLM-Based Imputation for Partially-Observed Time Series

## 🔍 What is LLM4IMP?

LLM4IMP is a prompt-based imputation model that uses a frozen large language model (LLM), such as GPT2, to fill missing values in partially observed multivariate time series. It reformulates imputation as a masked token completion task using patch embeddings, instructional prompts, and lightweight reprogramming adapters.

---

## 📦 High-Level Pipeline

```
X, missing_mask
   │
   ├──▶ Patch Embedding + Positional Encoding
   ├──▶ Prompt Embedding (instructional tokens)
   │
   └──▶ [Patch | Prompt] → Reprogramming Layer (Adapter)
                     │
             Frozen LLM (e.g. GPT2) ⛔ no finetuning
                     │
            Output Projection Head → reconstruction
                     │
          Combine with input to produce imputation
```

---

## 🧮 Mathematical Formulation

Let:
- \( X \in \mathbb{R}^{B \times T \times D} \): Input multivariate time series
- \( M \in \{0, 1\}^{B \times T \times D} \): Binary mask (1 = observed, 0 = missing)
- \( Z = [\text{PatchEmbed}(X), \text{Prompt}] \): Concatenated patch and prompt vectors
- \( H = \text{Reprogram}(Z) \): Reprogramming Layer output
- \( \hat{X} = \text{LLM}(H) \): Reconstructed values
- \( \tilde{X} = M \cdot X + (1 - M) \cdot \hat{X} \): Final imputed sequence

---

## 🧱 Architecture in `core.py`

### Class: `_LLM4IMP(ModelCore)`

### `__init__()`:
- Hyperparameters: `n_steps`, `n_features`, `patch_size`, `d_model`, `d_ffn`, `d_llm`, `n_heads`, `dropout`, etc.
- Modules used:
  - `PatchEmbedding`: Converts raw time series to token sequences.
  - `PromptBuilder`: Instructional prompt text → embedding.
  - `ReprogrammingLayer`: Adapter layer to inject structure.
  - `Frozen LLM`: GPT2 encoder from HuggingFace.
  - `FlattenHead`: Final projection to recover shape.

### `forward()`:
Input: dict with `X`, `missing_mask`, optionally `X_ori`, `indicating_mask`.
Steps:
1. Normalize and patch `X`
2. Build prompt tokens
3. Reprogram: `[patch + prompt] → adapter`
4. Feed into frozen GPT2
5. Project back to time-series space
6. Replace missing values only
7. Return output dict with `imputation`, `reconstruction`, and optionally `loss`/`metric`

---

## 💡 Key Innovations

| Component             | Innovation                                                   |
|----------------------|---------------------------------------------------------------|
| Prompt-based Input   | Encodes missing context into instruction-style text prompts   |
| Frozen LLM           | Uses language modeling to reason over temporal gaps           |
| Reprogramming Layer  | Efficient injection of time-series structure                  |
| Mask-aware Decoding  | Imputes only missing values with original preserved           |
| Generalizability     | Works across domains: energy, health, climate, etc.           |

---

## ✅ Template Compliance

- Subclassed from `ModelCore`
- Forward input/output are dict-based
- Integrated with PyPOTS trainer using:
  - `training_loss(...)`
  - `validation_metric(...)`
- Supports masked loss: `loss(X_recon, X_ori, indicating_mask)`

---

## 📌 To Implement/Check

- [x] `BackboneLLM4IMP` in `layers`
- [x] `_LLM4IMP` model in `core.py`
- [x] Public class `LLM4IMP` in `model.py`
- [x] Ready for PyPOTS validation/training pipeline
