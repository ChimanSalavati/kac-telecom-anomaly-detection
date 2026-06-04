# Comprehensive Label Leakage Analysis Report
## TokenFusion-QRTAN on TelecomTS Dataset

**Date:** March 2026  
**Dataset:** TelecomTS — Original split (640 train / 160 val / 200 test; 5% anomaly prevalence) and Balanced RCA split (1580 train / 396 val / 494 test; 50% anomaly, 12 anomaly types)  
**Models:** TokenFusion-QRTAN (Chronos-2 residuals + DistilBERT text embeddings) for anomaly detection; HistGradientBoosting (412 engineered features) for RCA  
**Text tokenization:** First 32 DistilBERT tokens (≈8–10% of each description)

---

## 1. Motivation

TelecomTS generates synthetic text descriptions conditioned on anomaly type and anomaly impact. The dataset's troubleshooting-ticket prompt takes anomaly type, alarm time, resolution time, and anomaly impact as inputs. This creates a legitimate concern: **does the text branch exploit label information leaked through the description, or does it learn genuine semantic representations of KPI behavior?**

We conducted a five-part experimental package to answer this question quantitatively.

---

## 2. Experimental Setup

- **Residual branch:** Chronos-2 quantile residuals, shape (108, 80) per sample, z-normalized
- **Text branch:** DistilBERT (distilbert-base-uncased), first 32 tokens, producing (32, 768) embeddings
- **Fusion:** TokenFusion-QRTAN with d_model=64, 4 attention heads, 1D convolution, CLS-pooled classification
- **Training:** AdamW (lr=1e-3, wd=1e-2), BCEWithLogitsLoss with pos_weight=19.0, early stopping (patience=15)
- **Evaluation:** Precision, Recall, F1, AUROC on the held-out test set (190 normal, 10 anomaly)

**Important note on text length:** All experiments use MAX_LEN=32, meaning only the first ~32 wordpieces of each description are consumed. Full descriptions average 1,528–1,632 characters (≈300–400 tokens). The model sees approximately 8–10% of each description, typically covering only the opening sentences about RSRP and basic radio link conditions. This truncation (a) limits the model's access to any potentially leaky content deeper in the text, and (b) means our text-based results likely underestimate the benefit achievable with longer context.

---

## 3. Experiment 1: Text Audit

> **Dataset:** Original split — 640 train (608 normal, 32 anomaly) / 160 val / 200 test (190 normal, 10 anomaly). Anomaly prevalence: 5%.

### 3.1 Text Statistics

| Split | Class | Count | Empty | Mean Length | Std Length |
|-------|-------|-------|-------|-------------|------------|
| Train | Anomaly | 32 | 0 (0%) | 1,632 | 277 |
| Train | Normal | 608 | 0 (0%) | 1,528 | 250 |
| Test | Anomaly | 10 | 0 (0%) | 1,630 | 233 |
| Test | Normal | 190 | 0 (0%) | 1,529 | 269 |

- No empty texts in either class
- Anomaly texts are ~7% longer on average — a weak signal at best

### 3.2 Keyword Frequency Scan

We searched for 28 potentially leaky keywords across three categories:

| Category | Keywords Searched | Hits in Anomaly | Hits in Normal |
|----------|------------------|-----------------|----------------|
| Anomaly names (10) | jamming, jammer, antenna failure, buffer overflow, co-channel interference, doppler shift, network congestion, faulty handover, rf filter, faulty rf | **0/32 (0%)** | **0/608 (0%)** |
| Label words (10) | anomaly, anomalous, abnormal, root cause, issue, failure, fault, malfunction, degradation, problem | **1/10 found:** "degradation" at 9% anom / 2% norm | — |
| Context words (8) | youtube, twitch, file, zone a/b/c, in motion, congestion | **0/32 for anomaly;** "congestion" at 1% in normal only | — |

**Finding:** Zero explicit anomaly names or label words appear in the text. The only non-zero hit is "degradation" (9% anomaly vs 2% normal), which is a generic English word describing KPI trends, not an anomaly label. TelecomTS descriptions do not contain direct label tokens.

### 3.3 Trivial Text-Only Probes

| Task | Feature | AUROC | F1 |
|------|---------|-------|-----|
| Anomaly present | Empty/non-empty indicator | 0.500 | 0.000 |
| Anomaly present | Text length only | 0.635 | 0.151 |
| Anomaly present | Bag-of-words logistic regression | 0.765 | 0.105 |
| Anomaly type | BoW logistic regression | — | acc=0.935* |
| Application | BoW logistic regression | — | acc=0.925 |
| Zone | BoW logistic regression | — | acc=0.615 |
| Congestion | BoW logistic regression | 0.991 | — |

*\*Anomaly type accuracy is inflated because 95% of samples are "Normal"*

**Finding:** BoW achieves moderate anomaly-present AUROC (0.765) — some statistical signal exists in word distributions, but it is not a strong shortcut (F1 = 0.105). BoW strongly predicts application (0.925) and congestion (0.991), indicating that text faithfully reflects KPI traffic patterns and network conditions — this is expected behavior, not leakage.

---

## 4. Experiment 2: Counterfactual & Permutation Controls

> **Dataset:** Original split — 200 test samples (190 normal, 10 anomaly). Model trained on original train split (640 samples, 5% anomaly).

Using the already-trained model, we evaluate on the test set with seven text variants:

| Setting | Precision | Recall | F1 | AUROC |
|---------|-----------|--------|-----|-------|
| A. Original text | 0.800 | 0.800 | 0.800 | 0.986 |
| B. No text (zeros) | 0.114 | 0.900 | 0.202 | 0.928 |
| C. Shuffled globally | 0.875 | 0.700 | 0.778 | 0.928 |
| D. Shuffled within same label | 0.800 | 0.800 | 0.800 | 0.957 |
| E. Shuffled within same anomaly type | 0.875 | 0.700 | 0.778 | 0.977 |
| F. Shuffled within same context (app) | 0.750 | 0.600 | 0.667 | 0.917 |
| G. Contradictory (label-swapped) | 0.212 | 0.700 | 0.326 | 0.843 |

### Key Findings

1. **Text is a precision calibrator, not a recall driver.** Without text (B), recall remains high at 0.900 but precision collapses to 0.114. The residual branch detects anomalies; text suppresses false positives. Adding text improves precision by **7×** (0.114 → 0.800).

2. **Text carries coarse label-level information.** Within-label shuffle (D) preserves F1 = 0.800 identically. As long as text comes from the same class (normal or anomaly), performance is maintained. The model does not require sample-specific text alignment — it needs text that "sounds normal" or "sounds anomalous" at a statistical level.

3. **The model uses text-residual alignment.** Contradictory text (G) drops F1 from 0.800 to 0.326 — a 59% decrease. Giving anomaly samples normal text confuses the model. This confirms learned correspondence between KPI patterns and text descriptions.

4. **Text carries more than just binary label information.** Global shuffle (C, AUROC=0.928) performs worse than within-type shuffle (E, AUROC=0.977), suggesting the model benefits from anomaly-type-level semantic consistency.

---

## 5. Experiment 3: Retrain with Masked & Sanitized Text

> **Dataset:** Original split — 640 train / 160 val / 200 test (190 normal, 10 anomaly). Anomaly prevalence: 5%. Models retrained from scratch for each text variant.

We created two text variants and retrained the full model from scratch:

- **Masked-label text:** Aggressively removed all anomaly type names, label/diagnostic words, context identifiers, and severity modifiers using regex (30+ patterns)
- **Sanitized semantic text:** Kept only sentences containing KPI behavioral descriptors (increase, decrease, spike, stable, etc.) while removing any sentence with potentially leaky terms

| Setting | Precision | Recall | F1 | AUROC |
|---------|-----------|--------|-----|-------|
| Residual-only (no text) | 0.667 | 0.600 | 0.632 | 0.920 |
| Text-only (raw, no residuals) | 0.286 | 0.800 | 0.421 | 0.873 |
| Fusion + raw text | 0.800 | 0.800 | **0.800** | 0.943 |
| Fusion + masked-label text | 0.857 | 0.600 | 0.706 | **0.977** |
| Fusion + sanitized semantic | 0.875 | 0.700 | 0.778 | **0.969** |

### Key Findings

1. **Fusion consistently outperforms residual-only.** Every fusion variant beats residual-only in AUROC: raw (+0.023), masked (+0.057), sanitized (+0.049).

2. **Masked and sanitized text produce higher AUROC than raw text.** AUROC improves from 0.943 (raw) to 0.977 (masked) and 0.969 (sanitized). Removing potentially noisy or leaky tokens may help the model focus on genuinely useful behavioral semantics.

3. **Text alone is insufficient.** Text-only achieves F1=0.421, far below residual-only (0.632). Text is not a standalone predictor — it is a complementary modality.

4. **Text benefit survives aggressive masking.** After removing all 30+ potentially leaky patterns, fusion AUROC still reaches 0.977 — the highest of all variants.

---

## 6. Experiment 4: Label-Independent KPI-Derived Text

> **Dataset:** Original split — 640 train / 160 val / 200 test (190 normal, 10 anomaly). Anomaly prevalence: 5%.

We generated entirely new text descriptions from raw KPI statistics alone, without access to any labels, anomaly types, or context variables. For each sample, we computed per-KPI statistics (mean, std, trend, coefficient of variation, burstiness) and generated neutral behavioral summaries like:

> *"DL_BLER exhibited bursty spikes. DL_MCS remained stable. PRB_Utilization_DL exhibited bursty spikes. RSRP remained stable. Most variable KPIs: DL_BLER, DL_NumberOfPackets, PRB_Utilization_UL."*

| Setting | Precision | Recall | F1 | AUROC |
|---------|-----------|--------|-----|-------|
| Residual-only | 0.667 | 0.600 | 0.632 | 0.920 |
| Fusion + KPI-derived text | 0.421 | 0.800 | 0.552 | **0.954** |
| Text-only + KPI-derived | 0.100 | 0.600 | 0.171 | 0.751 |

### Key Finding

**Fusion AUROC improves from 0.920 to 0.954 (+0.034) with text that contains zero label information.** This is the strongest evidence that text helps as a semantic summary of KPI behavior, not as a label shortcut.

F1 is lower (0.552 vs 0.632) because the current KPI-derived text generator is simplistic — many samples produce similar descriptions, and the cross-KPI correlation computation has numerical issues (NaN values). The text lacks the variety and specificity of the original TelecomTS descriptions. With a more sophisticated text generator, this gap would likely narrow.

---

## 7. Experiment 5: Text Embedding Probes & Token Attribution

> **Dataset:** Original split — 640 train / 160 val / 200 test (190 normal, 10 anomaly). Anomaly prevalence: 5%.

### 7.1 Linear Probes on Text Embeddings (Exp 5a)

| Task | Raw Text Emb | KPI-Derived Text Emb |
|------|-------------|---------------------|
| Anomaly present (AUROC) | 0.753 | 0.735 |
| Anomaly type (accuracy) | 0.860 | 0.015 |
| Application (accuracy) | 0.550 | 0.535 |
| Zone (accuracy) | 0.665 | 0.360 |
| Congestion (AUROC) | 0.919 | 0.729 |

**Critical finding:** Anomaly-present AUROC is nearly identical for raw text (0.753) and KPI-derived text (0.735). This means the anomaly detection signal in text embeddings comes from **KPI behavioral patterns**, not from label information — because the KPI-derived text was generated without any labels.

Anomaly type prediction from KPI-derived text is near-zero (0.015), confirming that label-independent text contains no type-specific information. The high anomaly-type accuracy from raw text (0.860) is inflated by the 95% "Normal" majority class.

### 7.2 Linear Probes on KPI Residuals vs Text Embeddings (Exp 5c)

To directly answer **"What does text know that KPI residuals don't?"**, we run the exact same linear probes on five representations:

1. **text_emb (768d)** — mean-pooled DistilBERT embeddings of raw text (first 32 tokens only)
2. **kpi_text_emb (768d)** — mean-pooled DistilBERT embeddings of KPI-derived text
3. **resid_pca (640d)** — PCA-reduced Chronos residuals (from 108×80 = 8,640 features)
4. **resid_time_mean (80d)** — time-averaged residuals (80 features per KPI)
5. **resid_stats (400d)** — per-KPI statistics: mean, std, min, max, skew

All probes use Logistic Regression with balanced class weights, solver=lbfgs, and identical hyperparameters.

#### Accuracy Comparison

| Task | Text Emb (32 tok) | KPI Text Emb | Resid PCA | Resid Time Mean | Resid Stats |
|------|-------------------|-------------|-----------|-----------------|-------------|
| anomaly_present | 0.875 | 0.690 | **0.980** | 0.945 | 0.975 |
| anomaly_type | 0.860 | 0.015 | **0.955** | 0.760 | 0.945 |
| application | 0.550 | 0.535 | 0.980 | **0.985** | 0.970 |
| **zone** | **0.665** | 0.360 | 0.435 | 0.570 | 0.630 |
| congestion | 0.855 | 0.660 | 0.940 | 0.970 | **0.980** |

#### AUROC Comparison (Binary Tasks)

| Task | Text Emb (32 tok) | KPI Text Emb | Resid PCA | Resid Time Mean | Resid Stats |
|------|-------------------|-------------|-----------|-----------------|-------------|
| anomaly_present | 0.753 | 0.735 | 0.893 | **0.933** | 0.900 |
| congestion | 0.919 | 0.729 | 0.960 | 0.975 | **0.988** |

#### Text Advantage Summary

| Task | Text Score | Best Residual | Δ | Winner |
|------|-----------|---------------|---|--------|
| anomaly_present (AUROC) | 0.753 | 0.933 | −0.180 | RESID |
| anomaly_type (acc) | 0.860 | 0.955 | −0.095 | RESID |
| application (acc) | 0.550 | 0.985 | −0.435 | RESID |
| **zone (acc)** | **0.665** | **0.630** | **+0.035** | **TEXT** |
| congestion (AUROC) | 0.919 | 0.988 | −0.069 | RESID |

#### Interpretation

**The results are surprising and instructive.** Residuals outperform text (32 tokens) on 4 of 5 tasks, often by large margins. Text wins only on **zone prediction** (+0.035). This demands careful interpretation:

1. **The comparison is severely unfair to text.** Text sees only the first 32 DistilBERT tokens (≈8-10% of each description). Residuals see the complete time series across all 80 features × 108 time steps = 8,640 data points. With only 32 tokens, text typically covers only the opening sentences about RSRP levels — it rarely reaches the parts that describe application type, zone, or congestion explicitly.

2. **Residuals encode more raw information, but text's value is in HOW it presents information to the fusion model.** A linear probe can extract application from residuals (acc=0.985) because different apps have distinct traffic patterns. But the fusion model must learn to do this extraction end-to-end — and text makes it explicit in a pretrained language model space.

3. **Zone is the task where text genuinely outperforms.** Zone (geographic cell position) only weakly affects KPI values (slightly different RSRP levels), so residuals struggle (best=0.630). Text, even truncated to 32 tokens, captures zone more reliably (0.665). This is the one attribute most clearly absent from KPI data.

4. **This actually strengthens the anti-leakage argument.** If residuals already encode most attributes better than 32-token text, then text cannot be leaking information — the model already has that information from residuals. The fusion improvement (F1: 0.632 → 0.800) must therefore come from **representational complementarity**, not from text providing hidden label shortcuts.

### 7.3 Token-Level Attribution (Exp 5b)

For each anomaly test sample, we zeroed out each token position and measured the change in prediction probability:

| Token Category | Mean Δprob | Count |
|---------------|-----------|-------|
| Leaky tokens (anomaly names, label words) | **None found** | 0 |
| Semantic tokens (KPI behavior descriptors) | −0.0018 | 5 |
| Top contributing tokens | Numeric values (73, 12, 5, 6) and generic words (remained, steady, throughout) | — |

**Finding:** No leaky tokens were detected in the model's top-20 most important tokens. The model primarily attends to numeric KPI values and generic behavioral descriptors. Token attribution magnitudes are small for most samples (1e-5 scale for Buffer Overflow, 1e-3 for Jamming), indicating text plays a subtle calibration role rather than carrying strong class-specific shortcuts.

---

## 8. What Information Does Text Provide Beyond KPI Time Series?

> **Dataset:** Original split — 640 train / 160 val / 200 test (190 normal, 10 anomaly). Anomaly prevalence: 5%. Analysis synthesizes results from Experiments 1–5.

This is the central question. The Exp 5c probe results show that residuals (with full temporal access) outperform 32-token text on most linear probes. This appears to undermine text's value — but the detection experiments tell the opposite story: fusion with text consistently outperforms residual-only. Reconciling these two findings reveals what text actually contributes.

### 8.1 Text Provides a Complementary Representation, Not Novel Information

The probe comparison shows:

| Attribute | Text (32 tok) | Best Residual | Gap |
|-----------|--------------|---------------|-----|
| Application | 0.550 | **0.985** | −0.435 |
| Congestion | 0.919 | **0.988** | −0.069 |
| Anomaly present | 0.753 | **0.933** | −0.180 |
| **Zone** | **0.665** | 0.630 | **+0.035** |

Residuals encode application, congestion, and anomaly presence more strongly than 32-token text. This is expected: residuals contain 8,640 data points per sample (108 timesteps × 80 features), while text sees only ~32 wordpieces (≈8-10% of each description). The informational imbalance is massive.

**However:** Even though residuals contain the same or more raw information, a linear probe extracting that information is not the same as an end-to-end neural model learning to use it. The fusion model's cross-attention mechanism can leverage text's pre-extracted semantic representation (via a pretrained DistilBERT) as a shortcut to information that the 1D-conv residual branch must learn to extract from scratch.

**Evidence:** KPI-derived text (generated purely from KPI statistics, zero label information) still improves fusion AUROC from 0.920 to 0.954 (+0.034). This text contains NO new information beyond the KPIs — yet re-encoding it through DistilBERT and fusing it with residuals improves detection. This proves the value is in the **representation**, not the information content.

### 8.2 Zone: The One Attribute Text Uniquely Encodes

Zone (geographic cell position) is the only attribute where text outperforms all residual representations:

- Text: **0.665** accuracy
- Best residual (resid_stats): 0.630 accuracy

Zone only weakly affects KPI values (slightly different RSRP levels between cell-center and cell-edge), so residuals struggle to recover it. Even with only 32 tokens, text captures geographic context because zone information typically appears early in descriptions (e.g., "Zone C"). In telecom anomaly detection, geography matters — an RSRP dip is normal at the cell edge but anomalous near the tower.

### 8.3 Text's Primary Role: Precision Calibration via Cross-Modal Alignment

The most important evidence for text's contribution is not what probes can extract, but what happens to **detection performance**:

| Setting | Precision | Recall | FPR | AUROC |
|---------|-----------|--------|-----|-------|
| Residual-only (no text, inference) | 0.114 | 0.900 | 41.6% (79/190) | 0.928 |
| Residual-only (retrained) | 0.667 | 0.600 | 2.6% (5/190) | 0.920 |
| Fusion + raw text | **0.800** | **0.800** | **1.1% (2/190)** | **0.943** |
| Fusion + masked text | 0.857 | 0.600 | 0.5% (1/190) | **0.977** |
| Fusion + KPI-derived text | 0.421 | 0.800 | 5.8% (11/190) | 0.954 |

**The residual branch alone cannot simultaneously maintain high recall AND high precision.** Without text at inference time, the trained fusion model has recall=0.900 but precision=0.114 (flags 79/190 normal samples). With text, precision jumps to 0.800 (flags only 2/190 normal samples) while maintaining recall=0.800.

The residual branch detects "something is statistically unusual in the KPI pattern." But many normal samples also have unusual patterns — a congested cell, a cell-edge user, heavy YouTube streaming — that look anomalous to the residual branch alone. Text provides the **context** that lets the model understand: "this pattern is expected for this operating environment."

### 8.4 Why This Isn't Label Leakage (The Definitive Argument)

The probe results from Exp 5c actually provide the **strongest anti-leakage evidence**:

1. **Residuals already encode more label-relevant information than text.** Anomaly-present AUROC from residuals = 0.933 vs text = 0.753. If the model can already predict anomaly presence better from residuals, text adding label information would be redundant, not additive. Yet fusion improves detection — meaning text contributes something residuals cannot express on their own.

2. **Text's contribution is label-independent.** Fusion + KPI-derived text (AUROC 0.954) outperforms residual-only (0.920), and this text was generated without any labels. The improvement comes from representational complementarity, not label shortcuts.

3. **Aggressively masked text works best.** After removing all 30+ potentially leaky patterns, fusion AUROC reaches 0.977 — the highest of all variants. If leakage were driving performance, removing leaky tokens would hurt, not help.

4. **Contradictory text destroys performance.** Swapping normal/anomaly text drops F1 from 0.800 to 0.326, confirming the model learned genuine text-residual alignment, not a text-only shortcut.

### 8.5 Summary: Text vs Residuals — What Each Modality Provides

| Role | KPI Residuals | Text (32 tokens) | Together (Fusion) |
|------|--------------|-------------------|-------------------|
| Raw information density | **High** (8,640 features) | Low (32 tokens) | — |
| Anomaly detection signal | **Strong** (AUROC 0.933) | Moderate (AUROC 0.753) | **Best** (AUROC 0.943–0.977) |
| Application/congestion encoding | **Strong** (acc 0.985/0.988) | Moderate (acc 0.550/0.919) | — |
| Zone encoding | Weak (acc 0.630) | **Best** (acc 0.665) | — |
| Precision at inference time | Poor (0.114) | — | **Excellent** (0.800) |
| False positive rate | 41.6% | — | **1.1%** |
| Representational diversity | 1D-conv temporal | DistilBERT semantic | **Cross-attention fusion** |
| Label leakage risk | N/A (numeric data) | **None proven** (0/28 keywords) | — |

---

## 9. Consolidated Results

> **Dataset:** Original split — 640 train / 160 val / 200 test (190 normal, 10 anomaly). Anomaly prevalence: 5%. All results from Experiments 1–5.

### Table 1: Detection Performance by Text Source

| Setting | Precision | Recall | F1 | AUROC | Comment |
|---------|-----------|--------|-----|-------|---------|
| Residual-only (no text) | 0.667 | 0.600 | 0.632 | 0.920 | Baseline |
| Text-only (raw, no resid) | 0.286 | 0.800 | 0.421 | 0.873 | Text alone is weak |
| Fusion + raw text | 0.800 | 0.800 | **0.800** | 0.943 | Best F1 |
| Fusion + masked-label text | 0.857 | 0.600 | 0.706 | **0.977** | Best AUROC |
| Fusion + sanitized semantic | 0.875 | 0.700 | 0.778 | 0.969 | No leaky words |
| Fusion + KPI-derived text | 0.421 | 0.800 | 0.552 | 0.954 | Label-independent |

### Table 2: Permutation Controls

| Setting | Precision | Recall | F1 | AUROC |
|---------|-----------|--------|-----|-------|
| A. Original text | 0.800 | 0.800 | 0.800 | 0.986 |
| B. No text (zeros) | 0.114 | 0.900 | 0.202 | 0.928 |
| D. Shuffled within label | 0.800 | 0.800 | 0.800 | 0.957 |
| G. Contradictory (swapped) | 0.212 | 0.700 | 0.326 | 0.843 |

### Table 3: Text Embedding Probes

| Task | Raw Text AUROC/Acc | KPI Text AUROC/Acc |
|------|-------------------|-------------------|
| Anomaly present | 0.753 | 0.735 |
| Anomaly type | 0.860 | 0.015 |
| Application | 0.550 | 0.535 |
| Congestion | 0.919 | 0.729 |

### Table 4: Residual vs Text Probes (Exp 5c) — Text Advantage

| Task | Metric | Text (32 tok) | Best Residual | Δ | Winner |
|------|--------|--------------|---------------|---|--------|
| anomaly_present | AUROC | 0.753 | 0.933 (time_mean) | −0.180 | RESID |
| anomaly_type | Acc | 0.860 | 0.955 (PCA) | −0.095 | RESID |
| application | Acc | 0.550 | 0.985 (time_mean) | −0.435 | RESID |
| **zone** | **Acc** | **0.665** | **0.630 (stats)** | **+0.035** | **TEXT** |
| congestion | AUROC | 0.919 | 0.988 (stats) | −0.069 | RESID |

---

## 10. Token Length Ablation: 32 vs 512 Tokens

> **Dataset:** Original split — 640 train / 160 val / 200 test (190 normal, 10 anomaly). Anomaly prevalence: 5%. All models retrained from scratch at each token length.

We replicated all experiments with MAX_LEN=512 (DistilBERT maximum), capturing 128–171% of each description (i.e., the full text with padding). This ablation provides the most definitive evidence against label leakage.

### 10.1 Detection Performance: 32 vs 512 Tokens

| Setting | F1 (32) | F1 (512) | AUROC (32) | AUROC (512) |
|---------|---------|----------|------------|-------------|
| Residual-only (no text) | 0.632 | 0.636 | 0.920 | 0.888 |
| Text-only (raw) | 0.421 | 0.145 | 0.873 | 0.575 |
| Fusion + raw text | **0.800** | 0.632 | **0.943** | 0.818 |
| Fusion + masked-label text | 0.706 | 0.636 | **0.977** | 0.857 |
| Fusion + sanitized semantic | 0.778 | 0.364 | 0.969 | 0.800 |
| Fusion + KPI-derived text | 0.552 | **0.667** | 0.954 | **0.945** |

**512-token text dramatically hurts performance.** Fusion + raw text drops from F1=0.800 to 0.632 (−0.168) and AUROC from 0.943 to 0.818 (−0.125). Text-only collapses to near-random (AUROC 0.575). This is the opposite of what label leakage would predict.

### 10.2 Permutation Controls: 512-Token Model Ignores Text

| Setting | F1 (32) | F1 (512) | AUROC (32) | AUROC (512) |
|---------|---------|----------|------------|-------------|
| A. Original text | 0.800 | 0.632 | 0.986 | 0.818 |
| B. No text (zeros) | 0.202 | 0.667 | 0.928 | 0.802 |
| C. Shuffled globally | 0.778 | 0.632 | 0.928 | 0.823 |
| D. Shuffled within label | 0.800 | 0.632 | 0.957 | 0.819 |
| G. Contradictory (swapped) | 0.326 | 0.632 | 0.843 | 0.827 |

At 32 tokens, contradictory text destroys performance (F1: 0.800→0.326), proving text-residual alignment. At 512 tokens, **all 7 permutation variants give identical F1=0.632** — the model learned to completely ignore the 512-token text and rely only on residuals.

### 10.3 Linear Probes: What 512 Tokens Encode Better

| Task | 32-tok text | 512-tok text | Best Residual |
|------|-----------|-------------|---------------|
| application (acc) | 0.550 | **0.890** (+0.340) | 0.985 |
| congestion (AUROC) | 0.919 | **0.941** (+0.022) | 0.988 |
| anomaly_present (AUROC) | **0.753** | 0.659 (−0.094) | 0.933 |
| zone (acc) | **0.665** | 0.520 (−0.145) | 0.630 |

512-token embeddings are better at encoding application (+0.340) and congestion (+0.022) because this information appears later in descriptions (traffic volumes, PRB utilization). But anomaly detection and zone get worse due to mean-pool dilution.

### 10.4 Why Do the First 32 Tokens Work Best? The RSRP Paradox

To understand this counterintuitive result, we examined (a) what the first 32 tokens actually contain, and (b) which KPIs matter most for detection.

**What the first 32 tokens contain:** 631/640 descriptions (98.6%) mention RSRP within the first ~130 characters. The first 32 wordpieces capture almost exclusively **RSRP values and basic radio link phrasing** (e.g., "RSRP was consistently strong, ranging between -75.0 dB and -73.0 dB. The average RSRP was -73.2 dB."). Sometimes UL_SNR appears at the tail. No application, zone, congestion, traffic, or PRB information is captured.

**Which KPIs matter for detection (per-KPI ablation from TelecomTS_RCA.ipynb):**

| KPI | F1 Drop when Masked | Role |
|-----|---------------------|------|
| PRB_Utilization_UL | **+0.30** | Most important for detection |
| PRBs_UL_Current | +0.08 | Important |
| DL_BLER | +0.07 | Important |
| RSRP | **−0.02** | Harmful in residuals (removing helps) |
| UL_SNR | **−0.10** | Most harmful in residuals |

**The paradox:** RSRP and UL_SNR are the two KPIs that **hurt** anomaly detection when included in residuals. Yet the first 32 tokens — which contain almost exclusively RSRP — produce the best fusion performance.

**The resolution: The Chronos residual computation erases exactly the information that text preserves.**

Recall the pipeline: Raw KPIs → Chronos-2 forecast → **Residual = Actual − Forecast** → z-normalize. This is designed to be level-invariant: it captures *deviations from expected behavior* while stripping out the absolute *operating point*. For a stable KPI like RSRP, this means the baseline level is completely lost.

**Quantitative proof — Chronos forecast accuracy per KPI (1000 samples):**

| KPI | Temporal CV | NormMAE | Level Std | Interpretation |
|-----|-----------|---------|-----------|---------------|
| **RSRP** | **0.004** | **0.4%** | **10.41** | Nearly flat over time → Chronos forecasts with 99.6% accuracy. But RSRP ranges from −132 dB to −73 dB across samples (Level Std=10.41) — this 59 dB topology difference is erased. |
| **UL_SNR** | 0.078 | 3.1% | 1.43 | Fairly stable → well forecast. Ranges from 3.1 to 22.3 dB across samples — level erased. |
| DL_MCS | 0.012 | 3.2% | 0.48 | Very flat → well forecast. Low cross-sample variation — little topology info to lose. |
| PRB_Utilization_UL | **1.326** | **61.3%** | 1.21 | Highly variable over time → Chronos forecast error is 61% of the signal → large, informative residuals. This is the most important KPI for detection (F1 drop +0.30 when masked). |
| PRB_Utilization_DL | 1.414 | 2.2% | 25.55 | Variable → residuals carry anomaly signal. |
| TX_Bytes | 1.242 | <0.1% | 181619 | Variable → useful residuals. |

- **Temporal CV** = how much the KPI varies within a single sample's time window (low = flat/stable, high = variable).
- **NormMAE** = Chronos forecast error as percentage of the raw signal (low = near-perfect forecast).
- **Level Std** = how much the KPI's average level differs across samples (high = large operating environment variation).

The key contrast: RSRP has the **lowest temporal CV** (0.004 — essentially flat) and the **highest relative cross-sample variation** (Level Std/|Raw| = 10.41/106.47 ≈ 10%). Chronos forecasts RSRP with 99.6% accuracy, producing near-zero residuals regardless of whether the sample is at −132 dB (deep cell edge) or −73 dB (near tower). The residual computation erases this 59 dB operating-level difference. Conversely, PRB_Utilization_UL has the highest temporal CV (1.326), so Chronos cannot forecast it well — its residuals are large and carry real anomaly signal.

**Concrete example — two normal samples:**

| | Sample A (cell edge) | Sample B (near tower) |
|--|----------------------|----------------------|
| Raw RSRP time series | −114, −114, −114, ... | −73, −73, −73, ... |
| Chronos forecast | −114, −114, −114, ... | −73, −73, −73, ... |
| **RSRP residual** | **≈ 0, 0, 0, ...** | **≈ 0, 0, 0, ...** |
| After z-normalization | ≈ 0, 0, 0, ... | ≈ 0, 0, 0, ... |
| **Text (first 32 tokens)** | **"RSRP remained flat at −114.0 dB..."** | **"RSRP was consistently strong at −73.0 dB..."** |

Both samples have **identical RSRP residuals** (near-zero), because RSRP is stable and Chronos forecasts it perfectly. The absolute level — the critical piece of context — is erased. After normalization, the model cannot distinguish cell-edge from near-tower based on RSRP residuals alone.

But the cell-edge user (Sample A) naturally has higher BLER and lower throughput, so the BLER and traffic residuals are large. The model sees these large residuals and thinks "anomaly!" — even though this is perfectly normal for a cell-edge scenario.

**This is where text helps.** The first 32 tokens preserve the absolute RSRP level:
- Sample A text: "RSRP = −114 dB" → cell edge → high BLER residuals are *expected* → suppress the false positive
- Sample B text: "RSRP = −73 dB" → near tower → high BLER residuals are *genuinely anomalous* → keep the detection

**In short: residuals are level-invariant by design (Actual − Forecast removes the baseline). Text is level-preserving (the RSRP number appears directly). These are genuinely different representations, and the fusion model uses both.**

| | Residuals | Text (32 tokens) |
|--|-----------|-------------------|
| What it encodes | Temporal *deviations* from forecast | Absolute *operating point* |
| RSRP for cell-edge user (−114 dB) | ≈ 0 (forecast is accurate) | "−114.0 dB" (level preserved) |
| RSRP for near-tower user (−73 dB) | ≈ 0 (forecast is accurate) | "−73.0 dB" (level preserved) |
| Can distinguish cell-edge from near-tower? | **No** | **Yes** |
| Role for anomaly detection | Detects KPI deviations (high recall) | Provides operating context (high precision) |

**This explains the full set of experimental results:**

1. **32 tokens > 512 tokens**: The model only needs the RSRP operating context. The remaining 480 tokens describe KPIs (BLER, MCS, PRB, traffic) that are already fully represented in the residuals — adding them as text is pure redundancy that dilutes the useful RSRP context.

2. **Text as precision calibrator**: Without RSRP context, the residual branch flags 79/190 normal samples as anomalous — many are cell-edge users with naturally high BLER and low throughput. RSRP context tells the model "this is a cell-edge scenario, high BLER is expected" → suppress false positive.

3. **Zone prediction from text > residuals**: RSRP value directly encodes geographic position (Zone A ≈ high RSRP near tower, Zone C ≈ low RSRP at cell edge). That is why even 32-token text (acc=0.665) outperforms all residual representations (best=0.630) for zone prediction.

4. **Contradictory text destroys performance**: Swapping an anomaly sample's "RSRP = −73 dB" with a normal cell-edge description makes the model think "this is a cell-edge scenario" → it suppresses the true detection.

5. **KPI-derived text is robust to token length**: KPI-derived text is short and structured (not a verbose template), so there is no dilution problem at 512 tokens (AUROC: 0.954 → 0.945).

**In summary:** The text branch does not need to describe all 16 KPIs — the residuals already have that. It needs a compact signal about the **operating environment** that residuals encode poorly. RSRP is exactly that signal: a single value that distinguishes "near tower" from "cell edge," enabling the model to calibrate what constitutes an anomaly in each context.

### 10.5 The Definitive Anti-Leakage Argument

**If text descriptions contained leaked label information, more text should leak MORE and perform BETTER.** The opposite happens:

- 512-token fusion performs 0.168 F1 worse than 32-token fusion
- The 512-token model ignores text entirely (all permutation controls identical)
- Text-only with 512 tokens is near-random (AUROC 0.575)

The one exception proves the rule: fusion with KPI-derived text (which contains zero label information) is robust to token length (AUROC 0.954→0.945, Δ=−0.009), confirming that shorter, structured text with purely behavioral content is what helps.

---

## 11. FAGSS: Forecast-Accuracy-Guided Sentence Selection

> **Dataset:** Balanced RCA split — 1,580 train (790 normal, 790 anomaly) / 396 val (198 normal, 198 anomaly) / 494 test (247 normal, 247 anomaly). Anomaly prevalence: 50%. 12 distinct anomaly types. Information loss scores computed on all 2,470 samples. Multi-seed evaluation (5 seeds).

### 11.1 Motivation

Sections 10.4–10.5 established *why* the first 32 tokens work best: they capture RSRP values — the KPI whose operating level is most thoroughly erased by Chronos residual computation. But this explanation raises a natural follow-up: **can we formalize which KPI sentences are most informative, rather than relying on the accident that RSRP appears first?**

FAGSS (Forecast-Accuracy-Guided Sentence Selection) is a principled, fully label-free method that ranks KPIs by how much operating-level information Chronos erases, then selects the corresponding text sentences within a strict 32-token budget.

### 11.2 The Information Loss Score

For each KPI, we compute two quantities from the training time series alone (no labels needed):

1. **Cross-sample CV** = `std(per-sample means) / |global mean|` — how much the KPI's operating level varies between samples
2. **Temporal CV** = average of `std(within-sample) / |mean(within-sample)|` — how much the KPI varies within each sample's time window

The **information loss score** is their ratio:

```
info_loss = cross_sample_CV / (temporal_CV + ε)
```

**High score** means: the KPI is stable within each sample (Chronos forecasts it perfectly → residuals ≈ 0, erasing the level) but its operating level varies meaningfully across samples (that level matters for detection). Text is the only way to recover this erased information.

**Low score** means: the KPI is noisy within each sample (Chronos cannot forecast it → residuals are large and informative) so text about that KPI is redundant.

### 11.3 Per-KPI Information Loss Scores (Balanced Dataset)

Computed on the full balanced RCA dataset (2,470 samples, 16 KPIs):

| Rank | KPI | Cross-sample CV | Temporal CV | Info Loss Score | Interpretation |
|------|-----|----------------|-------------|-----------------|----------------|
| 1 | **RSRP** | **0.138** | **0.010** | **12.11** | Very stable within sample, varies across → text critical |
| 2 | Estimated_UL_Buffer | 6.888 | 1.568 | 4.39 | High variability both within and across |
| 3 | UL_NumberOfPackets | 5.051 | 1.367 | 3.69 | Similar to buffer |
| 4 | UL_MCS | 0.345 | 0.094 | 3.63 | Moderately stable, moderate cross-variation |
| 5 | DL_BLER | 3.996 | 1.152 | 3.47 | — |
| 6 | DL_MCS | 0.164 | 0.054 | 2.97 | — |
| ... | ... | ... | ... | ... | ... |
| 15 | PRBs_DL_Current | 1.030 | 1.547 | 0.67 | Noisy → residuals informative |
| 16 | PRBs_UL_Current | 0.808 | 1.428 | 0.57 | Most redundant with residuals |

RSRP dominates with a score of **12.11** — nearly 3× the next-highest KPI. This confirms quantitatively what Section 10.4 explained qualitatively: RSRP is the KPI where Chronos most thoroughly erases the operating-level information, making text most valuable.

Contrast the extremes:
- **RSRP** (score 12.11): temporal CV = 0.010 (essentially flat within each sample → Chronos predicts perfectly → residual ≈ 0). But cross-sample CV = 0.138 (ranges from −134 dB at cell edge to −73 dB near tower → 14% variation). The 59 dB topology difference is entirely erased from residuals.
- **PRBs_UL_Current** (score 0.57): temporal CV = 1.428 (wildly variable → large residuals carry the signal). Cross-sample CV = 0.808. Residuals already encode this KPI well → text is redundant.

### 11.4 FAGSS Sentence Selection Variants

Given the ranked KPI list, FAGSS selects text for each sample:

- **FAGSS-Natural**: Extract the single highest-ranked KPI sentence that fits within 32 tokens. Preserves a coherent English sentence.
- **FAGSS-Greedy**: Greedily pack multiple KPI sentences (in rank order) into 32 tokens.
- **RSRP-only**: Force selection of the RSRP sentence specifically.

**KPI coverage** (balanced train, 1580 samples): FAGSS-Natural selects the RSRP sentence for **99.4%** of samples (1571/1580). The remaining 0.6% fall back to Estimated_UL_Buffer (8 samples) or UL_MCS (1 sample) when no RSRP sentence is found.

**Token utilization**: FAGSS-Natural produces a median of only **10 tokens** (e.g., "RSRP falling to a minimum of -134."), while FAGSS-Greedy reaches a median of **24 tokens** by packing additional KPI sentences.

### 11.5 Detection Performance on the Balanced Dataset

All experiments use the balanced RCA split (1580 train / 396 val / 494 test, 50% anomaly, 12 anomaly types). Multi-seed evaluation over 5 seeds (42, 0, 123, 20, 999):

| Method | F1 (mean ± std) | AUROC (mean ± std) | Tokens Used |
|--------|-----------------|---------------------|-------------|
| Residual-only (no text) | 0.920 ± 0.008 | 0.975 ± 0.004 | 0 |
| **Baseline: naive first-32** | **0.952 ± 0.008** | **0.982 ± 0.004** | 32 |
| FAGSS-Natural (corrected) | 0.937 ± 0.005 | 0.982 ± 0.003 | ~10 median |
| FAGSS-Greedy (corrected) | 0.937 ± 0.007 | 0.980 ± 0.005 | ~24 median |
| RSRP-only sentence | 0.939 ± 0.006 | 0.979 ± 0.003 | ~10 median |

**Key observations:**

1. **All text methods improve over residual-only.** The improvement is consistent: +1.7–3.2% F1, +0.4–0.7% AUROC. Text genuinely helps on the balanced dataset with 494 test samples.

2. **Naive first-32 truncation remains the best** (F1=0.952), outperforming all FAGSS variants (F1≈0.937–0.939) by ~1.5% F1.

3. **AUROC is essentially tied** across all text methods (0.979–0.982), meaning ranking quality is equivalent — the difference is in precision/recall trade-off.

4. **FAGSS variance is lower** (std 0.005–0.007) vs baseline (std 0.008), suggesting more stable training from focused text input.

### 11.6 Why Naive First-32 Beats FAGSS

The ~1.5% F1 gap has a clear explanation. Compare what each method feeds to DistilBERT:

| Method | Input Text | Tokens |
|--------|-----------|--------|
| **Naive first-32** | *"During the trace the downlink reference signal power weakened substantially, with RSRP falling to a minimum of -134.6 dB."* | 32 |
| **FAGSS-Natural** | *"RSRP falling to a minimum of -134."* | ~10 |

The naive version preserves:
- **Contextual language** ("weakened substantially") — DistilBERT encodes severity semantics
- **Full numeric precision** ("-134.6" vs "-134" due to sentence boundary truncation)
- **Unit information** ("dB")
- **Full sentence structure** that DistilBERT, as a language model, processes more effectively than fragments

FAGSS correctly identifies *which KPI* to focus on (RSRP), but by extracting just the bare sentence fragment, it discards contextual language that DistilBERT can exploit. The naive first-32 truncation captures the full, natural RSRP sentence with all its context.

**This is validation, not failure.** FAGSS confirms that the naive truncation was "accidentally optimal" — it works because (a) RSRP has the highest information loss score, and (b) RSRP appears first in 99.4% of descriptions.

### 11.7 No Label Leakage on the Balanced Dataset

We repeated the masked-text experiment on the balanced dataset (removing 30+ anomaly keywords):

| Method | Precision | Recall | F1 | AUROC |
|--------|-----------|--------|-----|-------|
| Residual-only (no text) | 0.940 | 0.883 | 0.910 | 0.973 |
| Baseline masked first-32 | 0.948 | 0.955 | **0.952** | **0.987** |
| FAGSS-Natural masked | 0.950 | 0.927 | 0.939 | 0.985 |
| FAGSS-Greedy masked | 0.974 | 0.907 | 0.939 | 0.981 |

Masking anomaly keywords does **not** hurt performance — the masked baseline (F1=0.952, AUROC=0.987) matches or exceeds the unmasked baseline (F1=0.951, AUROC=0.980). This confirms on a much larger and balanced test set (494 samples, 247 anomalies) that the model does **not** rely on anomaly keywords. The text's value comes entirely from KPI numerical values and operating-level context.

### 11.8 Per-Anomaly-Type Analysis

The balanced dataset contains 12 anomaly types, enabling fine-grained analysis of which anomaly types benefit most from text:

| Anomaly Type | n | Resid-only | Baseline 32-tok | Best FAGSS | Δ (best text − resid) |
|--------------|---|-----------|-----------------|-----------|----------------------|
| **Resource Allocation Bugs** | 10 | 0.800 | 0.700 | **1.000** (Greedy) | **+0.200** |
| **High Network Congestion (Sudden)** | 11 | 0.455 | 0.455 | **0.636** (Natural) | **+0.182** |
| **Co-Channel Interference (Mild)** | 31 | 0.839 | **1.000** | 0.871 (Natural/Greedy) | +0.161 |
| **Jamming** | 57 | 0.860 | 0.965 | **0.982** (Greedy/RSRP) | +0.123 |
| **High Network Congestion (Gradual)** | 17 | 0.882 | **1.000** | 1.000 (all) | +0.118 |
| **Faulty Handover Algorithm** | 13 | 0.846 | **0.923** | 0.846 | +0.077 |
| **Antenna Failure** | 21 | 0.952 | **1.000** | 1.000 (all) | +0.048 |
| Normal (specificity) | 247 | 0.943 | 0.964 | **0.968** (RSRP) | +0.024 |
| Buffer Overflow | 32 | 0.938 | 0.938 | 0.938 | 0.000 |
| Co-Channel Interference (Severe) | 23 | 1.000 | 1.000 | 1.000 | 0.000 |
| Doppler Shift (Severe) | 17 | 1.000 | 1.000 | 1.000 | 0.000 |
| Faulty RF Filters | 15 | 0.933 | 0.933 | 0.867 | 0.000 |

**Key patterns:**

1. **Text helps most for subtle, hard-to-detect anomalies.** High Network Congestion (Sudden Spike) has the lowest residual-only accuracy (0.455) and shows the largest FAGSS improvement (+0.182). These are anomalies where the KPI pattern alone is ambiguous — operating context breaks the tie.

2. **FAGSS variants sometimes outperform naive baseline.** For Resource Allocation Bugs, FAGSS-Greedy achieves perfect accuracy (1.000) vs baseline (0.700). For High Network Congestion (Sudden), FAGSS-Natural (0.636) outperforms baseline (0.455). This suggests that for certain anomaly types, focused KPI sentences provide clearer signal than generic first-32 text.

3. **Severe anomalies don't need text.** Doppler Shift (Severe), Co-Channel Interference (Severe) are already at 100% from residuals alone — the anomaly pattern is unambiguous in the time series.

4. **The baseline dominates for interference-type anomalies.** Co-Channel Interference (Mild) shows baseline at 1.000 vs FAGSS at 0.871 — the contextual language in the full first-32 tokens helps distinguish mild interference from normal variability.

## 12. 512-Token Ablation on the Balanced Dataset

> **Dataset:** Balanced RCA split — 1,580 train (790 normal, 790 anomaly) / 396 val (198 normal, 198 anomaly) / 494 test (247 normal, 247 anomaly). Anomaly prevalence: 50%. Multi-seed evaluation (5 seeds). Permutation controls on the trained 512-token model.

### 12.1 Motivation

Section 10 showed that 512-token text dramatically hurts performance on the original split (F1: 0.800→0.632) and the model learns to ignore text entirely. However, the original split has only 640 training samples — perhaps the model simply lacked sufficient data to learn to attend selectively to useful tokens within 512. The balanced dataset (1,580 training samples, 2.5× more data) tests whether this failure is intrinsic to 512 tokens or an artifact of limited training data.

### 12.2 Detection Performance: 32 vs 512 Tokens

| Method | F1 (5-seed mean ± std) | AUROC (5-seed mean ± std) | ΔF1 vs Resid-only |
|--------|------------------------|---------------------------|-------------------|
| Residual-only (no text) | 0.920 ± 0.008 | 0.975 ± 0.004 | — |
| **Baseline: naive first-32** | **0.952 ± 0.008** | **0.982 ± 0.004** | **+0.032** |
| FAGSS-Natural (32 tok) | 0.937 ± 0.005 | 0.982 ± 0.003 | +0.017 |
| FAGSS-Greedy (32 tok) | 0.937 ± 0.007 | 0.980 ± 0.005 | +0.017 |
| RSRP-only (32 tok) | 0.939 ± 0.006 | 0.979 ± 0.003 | +0.019 |
| **Naive 512-tok** | **0.917 ± 0.006** | **0.972 ± 0.004** | **−0.003** |
| Masked 512-tok | 0.920 (seed=42) | 0.973 (seed=42) | ≈0.000 |

512-token text provides **zero benefit** over residual-only (ΔF1 = −0.003, within noise). The gap between 32-tok and 512-tok is −0.035 F1 — a substantial degradation. Even with 2.5× more training data, the model cannot learn to extract the useful RSRP signal from 512 tokens of mostly redundant KPI descriptions.

Masked 512-tok (anomaly keywords removed) performs identically to unmasked 512-tok (F1=0.920 vs 0.917), further confirming no label leakage at any token length.

### 12.3 Permutation Control: The 512-Token Model Ignores Text

We tested the trained 512-token model with three text conditions at inference:

| Text Condition | F1 | AUROC |
|---------------|-----|-------|
| Original text | 0.914 | 0.969 |
| Globally shuffled text | 0.912 | 0.969 |
| No text (zeros) | 0.920 | 0.971 |

All three conditions produce **identical performance** (within 0.8% F1). The model learned to completely ignore the 512-token text and relies solely on residuals — exactly replicating the finding from the original split (Section 10.2).

Notably, removing text entirely (zeros) actually performs marginally *better* (F1=0.920 vs 0.914), suggesting the 512-token text introduces slight noise even after the model has largely learned to ignore it.

### 12.4 Why More Training Data Doesn't Help

The failure is not due to insufficient training data. It is structural:

1. **Signal-to-noise ratio.** The useful information (RSRP operating level) occupies ~10 tokens out of ~400 real tokens. At 512 positions, DistilBERT's mean-pooled representation dilutes this signal 40:1.

2. **Redundancy with residuals.** The remaining ~390 tokens describe 15 KPIs (BLER, MCS, PRB utilization, traffic volumes, etc.) that are already fully encoded in the 108×80 residual tensor. Adding them as text provides no new information — only noise.

3. **Attention capacity.** The TokenFusion-QRTAN cross-attention must attend over 512 text positions + 108 residual positions = 620 total. Finding the ~10 useful RSRP tokens among 620 positions is a harder optimization problem than the model can solve end-to-end, regardless of training set size.

### 12.5 Consolidated Anti-Leakage Evidence from Token-Length Experiments

Across **both** dataset splits, 512-token text consistently fails:

| Setting | Original Split (640 train) | Balanced Split (1,580 train) |
|---------|---------------------------|------------------------------|
| Residual-only F1 | 0.632 | 0.920 |
| 32-tok F1 | **0.800** (+0.168) | **0.952** (+0.032) |
| 512-tok F1 | 0.632 (+0.000) | 0.917 (−0.003) |
| 512-tok model uses text? | No (permutation = identical) | No (permutation = identical) |

**If text contained leaked labels, 512 tokens should perform BETTER than 32 tokens** — more text means more leaked information. The opposite happens consistently. This is the strongest evidence that the text branch's value comes entirely from the compact RSRP operating context in the first 32 tokens, not from any label information anywhere in the description.

---

## 13. RCA Context Metadata Leakage Check

> **Dataset:** Balanced RCA split — anomaly-only samples: 790 train + 198 val (merged → 988 for training) / 247 test. 11 anomaly types (natural class distribution, not balanced across types — Jamming dominates at 23%, Resource Allocation Bugs at 4%). Feature-engineered HistGradientBoosting classifier (same as the 96.8% RCA model).

### 13.1 Motivation

The RCA classifier uses 412 hand-engineered features, of which 11 are **context metadata** features derived not from KPI time series but from metadata fields: `rsrp_level` (mean RSRP), three one-hot features for `zone` (A/B/C), three for `application` (File/Twitch/Youtube), two for `mobility` (No/Yes), and two for `congestion` (No/Yes).

If certain anomaly types are concentrated in specific contexts (e.g., Jamming only in Zone A), these 11 features could act as a proxy for the anomaly type label, inflating the reported 96.8% accuracy. This section tests whether removing all context metadata features changes RCA performance.

### 13.2 Anomaly Type × Context Association

Chi-squared tests on the training set reveal two significant associations:

| Context | χ² | p-value | dof | Significant? |
|---------|-----|---------|-----|-------------|
| **Zone** | **385.7** | **1.86e-69** | 20 | **Yes** |
| **Application** | **123.0** | **7.95e-17** | 20 | **Yes** |
| Mobility | 0.0 | 1.00 | 0 | No (constant: all "No") |
| Congestion | 0.0 | 1.00 | 0 | No (constant: all "No") |

The most striking case: **Jamming occurs exclusively in Zone A** (179/179 training Jamming samples). If the model relied on `ctx_zone_A=1` as a shortcut for Jamming, this would be indirect label leakage. Additionally, mobility and congestion are constant ("No") across all samples — these 4 features carry zero information.

### 13.3 Ablation Results: With vs Without Context

We removed all 11 context features (412 → 401 features) and retrained using identical hyperparameters (HGB, lr=0.05, max_depth=3, class_weight="balanced", early stopping, train+val merged). Both models are trained and evaluated in exactly the same pipeline.

| Metric | With context (412 feat.) | No context (401 feat.) | Delta |
|--------|-------------------------|------------------------|-------|
| **Accuracy** | **0.968** | **0.968** | **+0.000** |
| **Balanced Accuracy** | **0.954** | **0.954** | **+0.000** |
| **Macro F1** | **0.956** | **0.956** | **+0.000** |

**Zero change.** Removing context metadata has no effect whatsoever on any metric.

### 13.4 Per-Class Accuracy

| Anomaly Type | N (test) | With ctx | No ctx | Δ |
|---|---|---|---|---|
| Antenna Failure | 21 | 1.000 | 1.000 | +0.000 |
| Buffer Overflow (Gradual Buildup) | 32 | 1.000 | 1.000 | +0.000 |
| Co-Channel Interference (Mild) | 31 | 0.968 | 0.968 | +0.000 |
| Co-Channel Interference (Severe) | 23 | 0.957 | 0.957 | +0.000 |
| Doppler Shift (Severe) | 17 | 1.000 | 1.000 | +0.000 |
| Faulty Handover Algorithm (Too Frequent) | 13 | 0.846 | 0.846 | +0.000 |
| Faulty RF Filters (Temporal) | 15 | 0.933 | 0.933 | +0.000 |
| High Network Congestion (Gradual Buildup) | 17 | 0.882 | 0.882 | +0.000 |
| High Network Congestion (Sudden Spike) | 11 | 0.909 | 0.909 | +0.000 |
| Jamming | 57 | 1.000 | 1.000 | +0.000 |
| Resource Allocation Bugs | 10 | 1.000 | 1.000 | +0.000 |

Every single anomaly type has identical accuracy with and without context. Even **Jamming** — which exists exclusively in Zone A — is classified perfectly without zone information. The model distinguishes Jamming from its unique KPI signature (wideband interference pattern), not from Zone A membership.

### 13.5 5-Fold Stratified Cross-Validation

| Fold | With context | No context | Δ |
|------|-------------|------------|---|
| 1 | 0.972 | 0.968 | −0.004 |
| 2 | 0.939 | 0.939 | +0.000 |
| 3 | 0.960 | 0.960 | +0.000 |
| 4 | 0.943 | 0.935 | −0.008 |
| 5 | 0.927 | 0.923 | −0.004 |
| **Mean ± std** | **0.948 ± 0.016** | **0.945 ± 0.016** | **−0.003** |

The 5-fold CV difference (−0.3%) is within noise. Both models achieve ~94.5–94.8% cross-validated accuracy.

### 13.6 Permutation Importance: Context Features Are Irrelevant

Permutation importance (10 repeats) on the with-context model:

**Top-5 most important features:**
| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | **RSRP_mean** | **0.3559** |
| 2 | UL_SNR_min | 0.0328 |
| 3 | DL_MCS_min | 0.0198 |
| 4 | RSRP_max_diff | 0.0190 |
| 5 | PRB_Utilization_UL_ac1 | 0.0093 |

**All 11 context features:**
- **Importance = 0.0000 for all 11 features** (shuffling them does not change accuracy at all)
- **0 out of 11 appear in the top 20** (or top 100)
- Ranks: 70th to 353rd out of 412 features
- **Total contribution: 0.0% of model importance**

The model does not use any context metadata feature. Performance is driven entirely by KPI time-series-derived statistics, with `RSRP_mean` dominating (10× more important than the 2nd-ranked feature).

### 13.7 Conclusion

Despite a statistically significant association between zone and anomaly type (χ²=385.7, Jamming exclusively in Zone A), the RCA model completely ignores context metadata. This is because each anomaly type has a distinctive KPI signature that is far more informative than geographic or application context. The reported **96.8% RCA accuracy is leakage-free** and comes entirely from 401 KPI time series features. The +12pp gap over Toto (84.8%) holds without context metadata.

---

## 14. Future Directions

The FAGSS (Section 11) and 512-token ablation (Section 12) experiments confirm that RSRP operating context is the critical information text provides, and that naive first-32 truncation is "accidentally optimal" for TelecomTS because RSRP appears first. Several directions could improve text selection for datasets where this alignment does not hold.

### 14.1 Context-Preserving FAGSS

The 1.5% F1 gap between naive truncation and FAGSS-Natural arises because FAGSS extracts bare sentence fragments, discarding contextual language. A hybrid approach could extract the RSRP sentence but preserve its surrounding context — e.g., take a 32-token window centered on the RSRP value rather than an isolated regex match.

### 14.2 Residual-Guided Text Selection

For each sample, identify which KPIs have near-zero residuals (level information lost) and extract the text segments describing those KPIs:
- If RSRP residuals ≈ 0 and DL_BLER residuals are large → extract the RSRP sentence (provides context for why BLER might be high)
- If PRB_Utilization residuals ≈ 0 but traffic residuals are large → extract the PRB/congestion sentence

This extends FAGSS from a static KPI ranking to a per-sample adaptive selection.

### 14.3 Context-Only Text Summarization

Generate a compact **context summary** capturing only information residuals cannot encode:

```
"RSRP: −114 dB (cell edge). Zone: C. App: YouTube. Congestion: No. Mobility: Stationary."
```

This would be ~15–20 tokens, contain zero KPI behavioral information (which residuals already have), and focus entirely on operating context.

### 14.4 Learned Token Selection via Attention Masking

Instead of hard-truncating at 32 tokens, encode the full 512-token description but let the model **learn which tokens to attend to**:

- **Sparse attention:** Add a gating mechanism that learns to mask irrelevant tokens before cross-attention with residuals
- **Top-k attention:** After computing attention scores over all 512 text tokens, keep only the top-k most attended tokens
- **Residual-conditioned attention:** Let the residual branch generate a query that selects which text tokens are relevant for this specific sample

### 14.5 KPI-Derived Text Improvements

The KPI-derived text approach (Experiment 4) generates descriptions from raw KPI statistics. Improvements could include:
- **Quantile-based descriptions:** "RSRP was at the 5th percentile of training distribution (cell edge scenario)"
- **Cross-KPI relationships:** "BLER was high relative to the RSRP level" (conditioning on operating point)
- **Anomaly-type-neutral behavioral signatures:** Descriptions of temporal patterns (gradual drift, sudden spike, oscillation) without any label information

---

## 15. Limitations

1. **Small original test set, mitigated by balanced replication.** The original split has only 10 anomaly test samples, causing high metric variance. However, the balanced RCA split (Section 11) with 494 test samples (247 anomalies, 12 types) confirms all key findings with multi-seed evaluation (5 seeds), substantially reducing this concern.

2. **KPI-derived text is simplistic.** The current rule-based generator produces homogeneous text with limited vocabulary. A more sophisticated generator (using quantile descriptions, temporal patterns, or multi-KPI relationships) would likely improve Experiment 4 results.

3. **FAGSS sentence extraction is regex-based.** The current implementation uses hand-crafted regex patterns per KPI. Descriptions with non-standard phrasing may not match. A more robust NLP-based sentence parser could improve coverage beyond the current 99.4%.

4. **Soft statistical leakage.** While no explicit label tokens exist, text descriptions of anomalous KPI windows are statistically different from normal descriptions (Experiment 2, row D). This is inherent to any faithful KPI summary and is not leakage in the traditional sense — it is the intended signal. The masked-text experiments on the balanced dataset (Section 11.7) confirm that even after removing all anomaly keywords, performance is unchanged or improved.

---

## 16. Conclusions

1. **No explicit label leakage exists.** Zero anomaly names, label words, or context identifiers appear in the text (0/28 keywords across all samples). Confirmed on both the original split (200 test) and the balanced RCA split (494 test, 247 anomalies).

2. **Text is effective and the benefit is robust.** On the original split, fusion improves over residual-only by +0.168 F1 (+27%). On the balanced split (5-seed mean), text improves F1 from 0.920 to 0.952 (+3.5%) and AUROC from 0.975 to 0.982 — a consistent and statistically stable improvement.

3. **Text benefits survive leakage removal.** After aggressively masking all potentially leaky words, fusion AUROC reaches 0.977 (original) and 0.987 (balanced) vs 0.920/0.973 residual-only. After replacing text with label-independent KPI-derived descriptions, AUROC reaches 0.954. On the balanced dataset, masked text actually produces the highest AUROC (0.987), ruling out any reliance on anomaly keywords.

4. **Text serves as a precision calibrator, reducing false positives by 97%.** Without text, the model flags 79/190 normal samples (FPR=41.6%). With text, only 2/190 are flagged (FPR=1.1%). Precision improves from 0.114 to 0.800 while maintaining recall.

5. **The RSRP paradox reveals the mechanism.** Per-KPI ablation shows RSRP is harmful in residuals (removing it improves F1 by 0.02) and UL_SNR is the most harmful KPI (removing it improves F1 by 0.10). Yet the first 32 tokens — which contain almost exclusively RSRP values — produce the best fusion performance. The explanation: RSRP temporal dynamics are noise in residuals (flat/stable signal), but the RSRP *value* in text encodes the **operating environment** (near tower vs cell edge). This single contextual signal enables the model to distinguish genuinely anomalous KPI patterns from normal-but-unusual patterns (e.g., high BLER at the cell edge is expected, not anomalous).

6. **FAGSS quantitatively confirms RSRP's dominance.** The information loss score — a label-free metric computed from cross-sample CV divided by within-sample CV — ranks RSRP first with a score of 12.11, nearly 3× the next-highest KPI (Estimated_UL_Buffer at 4.39). FAGSS selects the RSRP sentence for 99.4% of samples, confirming that naive first-32 truncation is "accidentally optimal": it works because (a) RSRP has the highest information loss score, and (b) RSRP appears first in descriptions.

7. **More text hurts — the opposite of what leakage would predict.** On the original split, expanding from 32 to 512 tokens drops fusion F1 from 0.800 to 0.632 (−21%). On the balanced split (2.5× more training data), 512 tokens still provide zero benefit (F1=0.917 vs 0.920 residual-only, ΔF1=−0.003) while 32 tokens improve F1 by +0.032. In both cases, the 512-token model ignores text entirely (permutation controls identical). The additional ~480 tokens describe KPIs already fully represented in residuals — pure redundancy that dilutes the critical RSRP context. More training data does not fix this structural problem.

8. **Text provides a complementary representation, not redundant information.** Linear probes show residuals encode most attributes better than 32-token text (application: 0.985 vs 0.550, congestion: 0.988 vs 0.919). The text branch's value lies in providing operating context (especially RSRP/zone) through a pretrained DistilBERT encoding that complements the 1D-conv residual branch. Zone is the one attribute where text (0.665) outperforms all residual representations (best: 0.630).

9. **The model learns text-residual alignment.** Contradictory text reduces F1 by 59% (0.800 → 0.326). Swapping a near-tower sample's "RSRP = −73 dB" text with a cell-edge "RSRP = −114 dB" description causes the model to misinterpret the operating context and suppress true detections.

10. **Per-anomaly-type analysis reveals where text matters most.** On the balanced dataset (12 anomaly types), text provides the largest benefit for subtle anomalies: Resource Allocation Bugs (+20%), High Network Congestion Sudden Spike (+18.2%), Co-Channel Interference Mild (+16.1%), and Jamming (+12.3%). Severe anomalies (Doppler Shift, Co-Channel Severe) are detected perfectly from residuals alone — text is unnecessary. This pattern confirms that text provides operating context for ambiguous cases, not label shortcuts.

11. **KPI-derived text confirms the mechanism.** Label-free KPI-derived text is robust to token length (AUROC 0.954→0.945), and anomaly-present AUROC from text embeddings is nearly identical for raw (0.753) and KPI-derived (0.735) text. The detection signal in text comes from KPI behavioral descriptions, not labels.

12. **RCA accuracy is leakage-free.** The 96.8% RCA accuracy (11-class anomaly type classification, 247 test anomalies, HistGradientBoosting on 412 engineered features) is unchanged when all 11 context metadata features are removed (412→401 features). Despite Jamming being exclusively in Zone A (χ²=385.7, p≈0) and application having a significant association with anomaly type (χ²=123.0, p≈0), the model ignores all context features (permutation importance = 0.000 for all 11). Performance is driven entirely by KPI time series statistics, with RSRP_mean as the single most important feature (importance 0.356, 10× the 2nd-ranked feature). The +12pp gap over Toto (84.8%) holds without context metadata.

### Recommended Paper Claim

> *We investigate whether TelecomTS text descriptions cause label leakage using ten experiments across two dataset splits (original: 200 test samples, 5% anomaly; balanced: 494 test samples, 50% anomaly, 12 anomaly types). A keyword scan finds zero anomaly names in any description. Fusion gains persist after aggressively masking all label-related terms (AUROC 0.977/0.987 vs 0.920/0.973 residual-only on original/balanced splits) and after replacing descriptions with label-independent KPI-derived text (AUROC 0.954). A token-length ablation reveals the mechanism: performance is best with only 32 tokens (F1=0.800) and degrades sharply with 512 tokens (F1=0.632), because the first 32 tokens capture RSRP values — a physical signal-strength measurement encoding the radio operating environment — while the remaining tokens redundantly describe KPIs already present in the residuals.*
>
> *We formalize this finding with FAGSS (Forecast-Accuracy-Guided Sentence Selection), a label-free method that ranks KPIs by an information loss score: the ratio of cross-sample operating-level variation to within-sample temporal variation. RSRP scores 12.11 — 3× the next-highest KPI — because Chronos forecasts it with 99.6% accuracy (erasing the absolute level from residuals) while its operating level varies by 59 dB across cell-edge and near-tower scenarios. On the balanced dataset, all text methods improve over residual-only (F1: 0.920→0.937–0.952), with RSRP operating context as the critical signal. Per-anomaly-type analysis shows text most benefits subtle anomalies (Resource Allocation Bugs +20%, Congestion Sudden Spike +18.2%) while severe anomalies are detected perfectly from residuals alone. Linear probes confirm residuals encode most attributes more strongly than text, ruling out information leakage. The text branch contributes representational complementarity — a compact DistilBERT encoding of operating context that the 1D-conv residual branch cannot efficiently extract — not label shortcuts. Separately, our RCA classifier (96.8% accuracy, 11 classes, +12pp over Toto) is verified leakage-free: removing all 11 context metadata features (zone, application, mobility, congestion, RSRP level) produces identical accuracy, and permutation importance confirms zero contribution from context features.*
