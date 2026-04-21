# FluGuard AI — YAMNet Cough Detection Training Notebooks

These notebooks document the full training pipeline for the custom cough classifier
used in FluGuard AI's audio engine. Both run on Kaggle (GPU T4, ~5 min each).

---

## Notebooks

### `nb-yamnet-ft-v1.ipynb` — Baseline
- **Data**: CoughVid v3 (1000 cough) + ESC-50 (1000 non-cough)
- **Architecture**: YAMNet (frozen) → 1024-dim embedding → Dense(256)+BN+Dropout → Dense(64)+BN+Dropout → sigmoid
- **Result**: AUC 99.0%, F1 96.8%
- **Problem found**: speech (e.g. "hello") sometimes misclassified as cough — ESC-50 contains no human speech

### `nb-yamnet-ft-v2.ipynb` — **Deployed Model**
- **Data**: CoughVid v3 (1000) + ESC-50 (1000) + **LibriSpeech speech (500) [new]**
- **Fix**: LibriSpeech speech clips added as hard negatives, teaching the model that speech ≠ cough
- **Result**: AUC **99.6%**, F1 **97.4%**, Recall **99.33%**, Precision **96.75%**
- **Output**: `best_cough_classifier.keras` — the file in `backend/models/`

---

## How to Run on Kaggle

1. Open https://kaggle.com → New Notebook → upload the `.ipynb` file
2. Add these datasets via **+ Add Input**:
   - `orvile/coughvid-v3` (CoughVid v3)
   - `mmoreaux/environmental-sound-classification-50` (ESC-50)
   - `victorling/librispeech-clean` (v2 only)
   - YAMNet model: search `google/yamnet` in Models tab
3. Enable GPU accelerator (T4)
4. Run All — total time ~5 minutes

---

## Pipeline Summary

```
Audio files (wav/webm/flac)
    → librosa  (16 kHz mono, normalised)
    → YAMNet   (frozen TF Hub, 1024-dim frame embeddings, averaged)
    → Dense head (trained, 2 layers + BatchNorm + Dropout)
    → sigmoid output  →  cough probability ∈ [0, 1]
```

Threshold at runtime: **0.50** (optimised for Recall ≥ 99%)
