# Stage 3 — MIL Benchmark (All Arms)

**Notebook:** `all_arms_mil_benchmark.ipynb`  
**Input:** Cached tile features (`.npy`) extracted by each encoder  
**Output:** Per-arm predictions, ROC curves, benchmark summary CSV

## What this notebook does

Runs the full Attention MIL benchmark across all 10 encoder arms using a
**frozen-encoder paradigm**: each encoder extracts tile features once (cached
to `.npy`), then an Attention MIL head is trained on the cached features.
This isolates the encoder's representation quality as the only variable.

## Benchmark arms

| Arm | Encoder | Pretraining strategy |
|-----|---------|----------------------|
| A | ViT-Tiny (ours) | DINO on TCIA — task-specific SSL |
| B | UNI | DINOv2 on 100k+ pathology slides |
| C | CONCH | Vision-language (pathology + reports) |
| D | ViT-Small | Supervised ImageNet |
| E | Virchow2 | SSL on 3M pathology slides |
| F | Prov-GigaPath | SSL on 1.3B pathology tiles |
| G | H-optimus-0 | SSL — large-scale pathology |
| H | Phikon-v2 | SSL — large-scale pathology |
| I | UNI2-h | DINOv2 — large-scale pathology |
| J | Hibou-L | SSL — large-scale pathology |

## Attention MIL architecture

Based on Ilse et al. (2018) gated attention:

```
Input: (N_tiles, embed_dim)  →  Linear(256) + ReLU + Dropout(0.25)
       ↓
Gated attention: A_V = tanh(W_V·h),  A_U = sigmoid(W_U·h)
                 A   = softmax(w · (A_V ⊙ A_U))
       ↓
Aggregation: M = Σ(A_i · h_i)  →  shape (1, 256)
       ↓
Classifier: Linear(256→64) + ReLU + Dropout(0.25) + Linear(64→1)
       ↓
Output: sigmoid → P(BM+)
```

## Evaluation protocol

- **3 train-test splits × 3-fold cross-validation** per arm
- Stratified by BM+/BM− to preserve class balance
- Primary metric: slide-level AUROC
- Secondary metric: accuracy at Youden-optimal threshold

## Dataset (Zhou et al., J Pathol 2024)

| Split | BM+ | BM− | Total |
|-------|-----|-----|-------|
| Train/Val | 45 | 73 | 118 |
| Test | 20 | 20 | 40 |

Tile size: 224×224 px | Tiles per slide: 1000

## Run environment

**Kaggle** (dual-GPU P100 recommended for heavy encoders like H-optimus-0, UNI2-h)

- HuggingFace token required for gated models (UNI, CONCH, Virchow2, etc.)
- Set token as Kaggle Secret `HF_TOKEN` (Cell 3)
- Run Cell 0 once to install deps, restart, then run from Cell 1

## Output files (saved to `../4_results/`)

| File | Description |
|---|---|
| `arm_X_splitN_y_pred.npy` | Predicted probabilities for test split N |
| `arm_X_splitN_y_true.npy` | Ground truth labels for test split N |
| `benchmark_summary_full.csv` | Mean ± std AUC and accuracy for all arms |
| `all_arms_roc_curves.png` | ROC curves for all arms overlaid |
| `all_arms_bar_charts.png` | Bar chart of mean AUC per arm |
