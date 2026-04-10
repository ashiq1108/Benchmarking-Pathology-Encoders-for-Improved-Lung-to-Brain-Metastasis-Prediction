> **Training status:** 10 epochs completed — ongoing. 
> Final encoder not yet released.
# Stage 2 — Custom DINO Encoder Training (Arm A)

**Notebook:** `tcia_dino_vit_tiny.ipynb`  
**Input:** `.h5` tile files from Stage 1  
**Output:** `tcia_dino_encoder.pt` — frozen backbone ready for feature extraction

## What this notebook does

Trains a **ViT-Tiny/16** encoder from scratch using DINO self-supervised learning
on the 96 TCIA brain-met NSCLC WSIs (~388,681 Vahadane-normalized tiles).

This is **Arm A** in the benchmark — the task-specific SSL encoder trained entirely
on the target domain (brain-met NSCLC pathology), as opposed to large-scale
pathology foundation models (UNI, CONCH, Virchow2, etc.) used in other arms.

### Architecture

- **Backbone:** ViT-Tiny/16, warm-started from ImageNet-21k weights
- **DINO head:** 3-layer MLP → bottleneck → normalized last layer
- **Student/Teacher:** EMA momentum update (cosine schedule)
- **Output dim:** 192 (ViT-Tiny embedding size)

### Design choices

ViT-Tiny was chosen over ViT-Small because of the small dataset size (96 WSIs).
A larger model would overfit during SSL. The task-specific pretraining hypothesis
is that domain relevance compensates for model capacity.

### Augmentations

- Multi-crop (2 global + N local views)
- HED stain jitter (Tellez et al., TMI 2019) — perturbs H/E/DAB channels stochastically
- Standard geometric augmentations (flip, crop, color jitter)
- No Macenko normalization — tiles are already Vahadane normalized from Stage 1

## Training config

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Optimizer | AdamW |
| LR schedule | Warmup + cosine decay |
| Mixed precision | `torch.amp.autocast('cuda')` |
| Batch size | Chunk-based (slides loaded in chunks) |
| Checkpoint | Every epoch, resumable |

## Run environment

**Google Colab** (GPU runtime recommended — T4 or A100)

1. Mount Google Drive (Cell 1)
2. Set paths in Cell 2 config block
3. Run Cell 0 once to install deps, then restart runtime
4. Run Cell 1 onwards

## Output files

| File | Description |
|---|---|
| `dino_epoch_N.pt` | Full checkpoint per epoch (student + teacher + optimizer) |
| `dino_latest.pt` | Always the most recent epoch (for resuming) |
| `tcia_dino_encoder.pt` | **Final output** — backbone weights only, no head |
| `dino_loss_curve.png` | Training loss per epoch |

## Resuming training

Set `RESUME_CKPT` in the config to `dino_latest.pt` path — Cell 9 will
automatically restore student, teacher, optimizer, and epoch counter.

## Saving the final encoder (Cell 12)

Strips the DINO projection head — saves backbone only. The saved file contains:
- `backbone_state_dict` — loadable directly into `timm.create_model('vit_tiny_patch16_224.augreg_in21k', num_classes=0)`
- `arch`, `embed_dim`, `n_epochs_trained` metadata

## Sanity check (Cell 13)

Loads the saved encoder and verifies output shape is `(4, 192)`.
