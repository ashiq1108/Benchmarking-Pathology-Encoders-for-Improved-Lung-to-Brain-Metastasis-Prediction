# Pretraining Strategy Matters: Benchmarking WSI Encoders for Brain Metastasis Risk Prediction in NSCLC

**April 2026 | Computational Pathology**

This repository benchmarks the impact of encoder pretraining strategy on
predicting brain metastasis (BM) risk from H&E-stained whole-slide images (WSIs)
in Stage I–III NSCLC patients. The central question: does pretraining on
domain-specific data beat large-scale pathology foundation models?

---

       ## Pipeline overview

```
       Raw SVS slides (TCIA)
                 │
                 ▼
┌─────────────────────────────────┐
│  1. Preprocessing               │  tcia_preprocessing.ipynb
│  Otsu tissue mask → blur filter │  Local machine
│  → Vahadane stain norm → .h5    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  2. Encoder Training (Arm A)    │  tcia_dino_vit_tiny.ipynb
│  DINO SSL on TCIA tiles         │  Google Colab (GPU)
│  ViT-Tiny/16 → tcia_dino_       │
│  encoder.pt                     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  3. MIL Benchmark (All Arms)    │  all_arms_mil_benchmark.ipynb
│  Frozen encoder → cached .npy   │  Kaggle (dual GPU)
│  → Attention MIL → AUC          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  4. Results                     │
│  ROC curves, bar charts,        │
│  benchmark_summary_full.csv     │
└─────────────────────────────────┘
```

---

## Benchmark arms

| Arm | Encoder | Pretraining strategy |
|-----|---------|----------------------|
| **A** | **ViT-Tiny (ours)** | **DINO on TCIA — task-specific SSL** |
| B | UNI | DINOv2 on 100k+ pathology slides |
| C | CONCH | Vision-language (pathology + reports) |
| D | ViT-Small | Supervised ImageNet |
| E | Virchow2 | SSL on 3M pathology slides |
| F | Prov-GigaPath | SSL on 1.3B pathology tiles |
| G | H-optimus-0 | SSL — large-scale pathology |
| H | Phikon-v2 | SSL — large-scale pathology |
| I | UNI2-h | DINOv2 — large-scale pathology |
| J | Hibou-L | SSL — large-scale pathology |

---

## Dataset

Based on **Zhou et al., J Pathol 2024** — 158 NSCLC WSIs (H&E, Stage I–III,
5+ year follow-up).

| Split | BM+ | BM− | Total |
|-------|-----|-----|-------|
| Train/Val | 45 | 73 | 118 |
| Test | 20 | 20 | 40 |

- Tile size: 224×224 px at 20× magnification
- Tiles per slide: 1000
- Stain normalization: Vahadane (SPCN) using Zhou reference tile

> **Data access:** The WSI slides are available from TCIA. Raw slide files
> are not included in this repository.

---

## Attention MIL

Based on Ilse et al. (2018) gated attention pooling:

- Input projection: Linear(embed_dim → 256) + ReLU + Dropout(0.25)
- Gated attention: A_V = tanh(W_V·h), A_U = sigmoid(W_U·h), A = softmax(w·(A_V⊙A_U))
- Aggregation: M = Σ(A_i · h_i)
- Classifier: Linear(256→64) + ReLU + Dropout(0.25) + Linear(64→1)
- Evaluation: 3 train-test splits × 3-fold CV, primary metric = slide-level AUROC

---

## Repository structure

```
wsi-encoder-brain-metastasis/
├── 1_preprocessing/
│   ├── tcia_preprocessing.ipynb    ← SVS → H5 tiles (local machine)
│   └── README.md
├── 2_encoder_training/
│   ├── tcia_dino_vit_tiny.ipynb    ← DINO SSL training (Google Colab)
│   └── README.md
├── 3_mil_benchmark/
│   ├── all_arms_mil_benchmark.ipynb ← Full benchmark, all 10 arms (Kaggle)
│   └── README.md
├── 4_results/
│   ├── benchmark_summary_full.csv
│   ├── figures/
│   │   ├── all_arms_roc_curves.png
│   │   └── all_arms_bar_charts.png
│   ├── arm_*_split*_y_{pred,true}.npy
│   └── README.md
├── checkpoints/                     ← encoder weights (see below)
├── requirements.txt
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (Stage 2 and 3)
- OpenSlide system library (Stage 1 only)

```bash
# Ubuntu — install OpenSlide system lib first
sudo apt-get install openslide-tools libopenslide-dev

# Install Python deps
pip install -r requirements.txt

# PyTorch — install the version matching your CUDA from https://pytorch.org
# e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### HuggingFace token

Arms B, C, E, F, G, H, I, J require a HuggingFace token to download gated models.

- **Kaggle:** Add as a Secret named `HF_TOKEN`
- **Colab:** Add as a Colab Secret named `HF_TOKEN`
- **Local:** `export HF_TOKEN=your_token_here`

---

## Running the pipeline

### Stage 1 — Preprocessing (local machine)

Open `1_preprocessing/tcia_preprocessing.ipynb` and set paths in Cell 1:

```python
SVS_DIR    = "/path/to/your/svs/files"
OUTPUT_DIR = "/path/to/save/h5/files"
REF_TILE   = "/path/to/zhou_reference_tile.png"
```

Run cells in order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9

### Stage 2 — Encoder training (Google Colab, GPU runtime)

Open `2_encoder_training/tcia_dino_vit_tiny.ipynb`:

1. Run Cell 0 once to install deps, then restart runtime
2. Mount Google Drive (Cell 1)
3. Set paths in Cell 2 config block
4. Run cells in order from Cell 1

### Stage 3 — MIL Benchmark (Kaggle, GPU accelerator)

Open `3_mil_benchmark/all_arms_mil_benchmark.ipynb`:

1. Run Cell 0 once to install deps, then restart
2. Set `HF_TOKEN` secret in Kaggle settings
3. Set paths in Cell 1 config block
4. Run cells in order from Cell 1

---

## Encoder weights

> ⚠️ **Work in progress:** The encoder has completed 10/N epochs of DINO
> pretraining. Final weights will be uploaded here once training is complete.

To load it:

```python
import timm, torch

enc = timm.create_model('vit_tiny_patch16_224.augreg_in21k',
                         pretrained=False, num_classes=0)
ckpt = torch.load('tcia_dino_encoder.pt', map_location='cpu')
enc.load_state_dict(ckpt['backbone_state_dict'])
enc.eval()
# Output shape: (batch, 192)
```

---
## Citation

If you use this code or the trained encoder, please cite:

```bibtex
@misc{bm-encoder-benchmark-2026,
  title  = {Benchmarking Pathology Encoders for Improved
             Lung-to-Brain Metastasis Prediction},
  year   = {2026},
}
```

@dataset{chadha2025tcia,
  author    = {Chadha, S. and Sritharan, D. and Dolezal, D. and Chande, S. and
               Hager, T. and Bousabarah, K. and Aboian, M. and Chiang, V. and
               Lin, M. and Nguyen, D. and Aneja, S.},
  title     = {MR Imaging and Segmentations with Matched Brain Biopsy Pathology
               Slides from Patients with Brain Metastases from Primary Lung Cancer
               (Brain-Mets-Lung-MRI-Path-Segs)},
  year      = {2025},
  publisher = {The Cancer Imaging Archive},
  version   = {2},
  doi       = {10.7937/k0sm-y874},
  url       = {https://doi.org/10.7937/k0sm-y874}
}

@article{chadha2026scientificdata,
  author  = {Chadha, S. and Sritharan, D. V. and Dolezal, D. and Chande, S. and
             Hager, T. and Bousabarah, K. and Aboian, M. S. and Chiang, V. and
             Lin, M. and Nguyen, D. X. and Aneja, S.},
  title   = {Matched MRI, Segmentations, and Histopathologic Images of Brain
             Metastases from Primary Lung Cancer},
  journal = {Scientific Data},
  volume  = {13},
  number  = {1},
  year    = {2026},
  doi     = {10.1038/s41597-025-06353-2}
}

@article{Zhou24NSCLC,
  author  = {Zhou, Haowen and Watson, Mark and Bernadt, Cory T and Lin, Steven (Siyu) and
             Lin, Chieh-yu and Ritter, Jon H and Wein, Alexander and Mahler, Simon and
             Rawal, Sid and Govindan, Ramaswamy and Yang, Changhuei and Cote, Richard J},
  title   = {AI-guided histopathology predicts brain metastasis in lung cancer patients},
  journal = {The Journal of Pathology},
  volume  = {263},
  number  = {1},
  pages   = {89--98},
  year    = {2024},
  doi     = {10.1002/path.6263}
}

## References

**Dataset**
- Zhou et al. (2024) — NSCLC brain metastasis WSI dataset, *Journal of Pathology*

- **Pretraining data (TCIA — used for encoder training in Stage 2):**
- Chadha et al. (2025) — Brain-Mets-Lung-MRI-Path-Segs [dataset], The Cancer Imaging Archive. https://doi.org/10.7937/k0sm-y874
- Chadha et al. (2026) — Matched MRI, Segmentations, and Histopathologic Images of Brain Metastases from Primary Lung Cancer, Scientific Data. https://doi.org/10.1038/s41597-025-06353-2

**Attention MIL**
- Ilse et al. (2018) — Attention-based Deep Multiple Instance Learning, *ICML*

**DINO / SSL**
- Caron et al. (2021) — Emerging Properties in Self-Supervised Vision Transformers (DINO), *ICCV*

**Stain augmentation**
- Tellez et al. (2019) — Quantifying the effects of data augmentation in computational pathology (HED jitter), *TMI*

**Foundation models used as benchmark arms:**

- **UNI (Arm B):** Chen et al. (2024) — Towards a General-Purpose Foundation Model for Computational Pathology, *Nature Medicine*. https://doi.org/10.1038/s41591-024-02857-3

- **CONCH (Arm C):** Lu et al. (2024) — A Visual-Language Foundation Model for Computational Pathology, *Nature Medicine*. https://doi.org/10.1038/s41591-024-02856-4

- **Virchow2 (Arm E):** Zimmermann et al. (2024) — Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology, *arXiv:2408.00738*

- **Prov-GigaPath (Arm F):** Xu et al. (2024) — A Whole-Slide Foundation Model for Digital Pathology from Real-World Data, *Nature*. https://doi.org/10.1038/s41586-024-07441-w

- **H-optimus-0 (Arm G):** Vert (2024) — H-optimus-0, Bioptimus. https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0

- **Phikon-v2 (Arm H):** Filiot et al. (2024) — Phikon-v2: A Large and Public Feature Extractor for Biomarker Prediction, *arXiv:2409.09173*

- **UNI2-h (Arm I):** Chen et al. (2025) — UNI2: Towards a Universal Whole Slide Image Encoder. https://huggingface.co/MahmoodLab/UNI2-h
