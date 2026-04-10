# Results

All outputs from the MIL benchmark (Stage 3).

## Summary files

| File | Description |
|---|---|
| `benchmark_summary_full.csv` | Mean ± std AUC and accuracy across 3 splits for all 10 arms |
| `figures/all_arms_roc_curves.png` | ROC curves for all arms across all splits |
| `figures/all_arms_bar_charts.png` | Mean AUC bar chart with error bars |

## Per-arm prediction files

For each arm (A–J) and each split (0–2), two `.npy` files are saved:

- `arm_X_splitN_y_pred.npy` — model's predicted P(BM+) for each test slide
- `arm_X_splitN_y_true.npy` — ground truth binary labels (1=BM+, 0=BM−)

These can be used to recompute any metric (AUC, accuracy, F1, calibration, etc.)
without re-running training.

## Loading results

```python
import numpy as np
from sklearn.metrics import roc_auc_score

y_true = np.load('arm_A_split0_y_true.npy')
y_pred = np.load('arm_A_split0_y_pred.npy')
print(f'Arm A Split 0 AUC: {roc_auc_score(y_true, y_pred):.3f}')
```

## Note on large `.npy` files

Raw prediction arrays are small (~40 values each) and safe to commit.
Feature cache `.npy` files (shape: 1000 × embed_dim per slide) are
excluded via `.gitignore` — these must be regenerated locally.
