# Verification Generalizability Study: Setup and Goals

## Overview

We investigate whether **certified robustness generalizes out-of-distribution (OOD)**.
Specifically: if a model has high Certified Robust Accuracy (CRA) on its training distribution,
does it also verify well on OOD images that it was never trained on?

**Setup**:
We train multiple models with **similar clean accuracy but different
CRA**, then run the same verification across in-distribution and OOD test sets. If
CRA generalizes, models with higher in-distribution CRA should also verify better OOD —
independent of clean accuracy.

---

## Models

We train one plain model and several adversarially trained variants, targeting the same clean
accuracy across all models.

### MNIST-FC (fully-connected, 10 classes)
All four models achieve ~93% clean accuracy on MNIST:

| Model | Training | Clean Acc | AutoAttack (ε=4/255) |
|-------|----------|-----------|----------------------|
| Plain | Standard cross-entropy | 93.5% | 49.3% |
| Adv ε=0.01 | PGD-AT | 92.9% | 78.9% |
| Adv ε=0.03 | PGD-AT | 92.4% | 88.9% |
| Adv ε=0.05 | PGD-AT | 92.8% | 89.1% |

Adversarially trained models are fine-tuned from the plain checkpoint using PGD training
(maximizing cross-entropy loss over an ℓ∞ ball). `--target_acc` early stopping is used to
ensure clean accuracy stays matched across models.

### GTSRB-CNN (convolutional, 43 traffic sign classes)
All four models achieve ~90-91% clean accuracy on GTSRB:

| Model | Training | Clean Acc |
|-------|----------|-----------|
| Plain (TRADES ε=0.01) | TRADES, β=6.0 | 90.70% |
| Adv ε=0.01 | TRADES, β=6.0 | 90.84% |
| Adv ε=0.02 | TRADES, β=3.0 | 90.61% |
| Adv ε=0.03 | TRADES, β=1.0 | 90.93% |

GTSRB models use **TRADES loss** (`CE(clean) + β * KL(clean || adv)`) instead of plain PGD-AT,
as it better preserves clean accuracy on harder multi-class data. Models are fine-tuned from
the plain checkpoint with a cosine annealing LR scheduler over 60 epochs.

All models are stored without input normalization (raw [0,1] pixel values), matching what
Alpha-Beta-CROWN expects at verification time.

---

## Verification Properties

We use Alpha-Beta-CROWN as the verifier, with properties written in **VNNLIB format**.

Each property encodes a local robustness query for a single image: *"for all inputs within ε
of this image (ℓ∞ ball), does the model predict the correct class?"* Formally, for an image
with true label `k`:

- **Input constraints:** `img[i] - ε ≤ X_i ≤ img[i] + ε`, clipped to [0, 1]
- **Output constraint (negated):** `∃ j ≠ k such that Y_j ≥ Y_k`

The verifier tries to falsify robustness. If it returns UNSAT (no counterexample exists), the
property is **verified (safe)**. If it finds a counterexample, the property is **falsified
(unsafe)**. If neither is resolved within the timeout, it returns **unknown**.

Properties are generated at four epsilon levels: **ε ∈ {1/255, 2/255, 3/255, 4/255}**, with
samples balanced across classes. The **same fixed property set is used for all models** —
this is critical, as it makes CRA directly comparable across models.

---

## Test Distributions

We evaluate each model on three distributions using the same VNNLIB property structure:

### 1. In-Distribution
Standard test set from the training distribution:
- **MNIST test set** — 100 properties (10 per class)
- **GTSRB test set** — 215 properties (5 per class)

This gives us the baseline CRA for each model.

### 2. Structural OOD
A **different dataset with the same label space and input format**:
- **EMNIST-digits** for MNIST models — same 10 digit classes, same 28×28 grayscale format,
  but drawn from a different handwriting collection. Images are orientation-corrected (90°
  rotation + horizontal flip) to match MNIST.
- **BTSD (Belgian Traffic Sign Dataset)** for GTSRB models — real traffic signs from Belgium,
  same 43-class mapping.

This tests whether certified robustness transfers to a related but shifted distribution.

### 3. Corruption OOD (PixMix)
**The original training-distribution images corrupted with PixMix**, which blends images with
fractal textures and mixing augmentations to produce visually degraded but still-recognizable
inputs:
- **MNIST + PixMix** — 100 properties (10 per class)
- **GTSRB + PixMix** — 420 properties (10 per class)

For PixMix properties, we apply a **consensus filter**: a sample is only included if all ONNX
reference models agree on its predicted label. This controls for images that are already
ambiguous or misclassified before any perturbation, keeping the focus on robustness rather
than accuracy.

---

## What We Want to See

The central question is: **does CRA rank-order models consistently across distributions?**

Concretely, we are looking for:

1. **Monotonicity.** If model A has higher CRA than model B on the in-distribution test set,
   does it also have higher CRA on both OOD sets? If so, CRA generalizes as a property.

2. **Magnitude of transfer.** CRA will drop OOD — the question is by how much. A large
   drop that erases the ranking between models would suggest CRA is highly distribution-specific.
   A proportional drop that preserves ordering would suggest it reflects something more intrinsic
   about the model.

3. **Difference between OOD types.** Structural OOD (EMNIST/BTSD) and corruption OOD (PixMix)
   represent different kinds of distribution shift. We expect PixMix to be harder (lower absolute
   CRA) but it is an open question which shift is more disruptive to the CRA ordering.

4. **Clean accuracy as a confounder.** Because all models are matched on clean accuracy, any
   differences in OOD CRA cannot be attributed to differences in baseline performance — they
   must reflect differences in the robustness structure of the models.

---

## Preliminary Results (MNIST-FC)

CRA (number of verified instances out of 100):

| Model | In-dist ε=4/255 | EMNIST ε=4/255 | PixMix ε=4/255 |
|-------|----------------|----------------|----------------|
| Plain | 56 | 26 | 31 |
| Adv ε=0.01 | 84 | 39 | 47 |
| Adv ε=0.03 | 92 | 58 | 70 |
| Adv ε=0.05 | 93 | 53 | 70 |

The ranking is preserved across all three distributions: models with higher in-distribution CRA
also verify better OOD. The absolute drop is large (roughly half of in-distribution CRA on
structural OOD), but the ordering is stable. PixMix shows a clearer separation between the
plain model and the adversarially trained ones than EMNIST does.

The GTSRB experiments are in progress.
