# Melanoma-AI-pipeline

# Sequential Melanoma Classification System

This repository contains the code and trained models for a sequential deep learning system for skin lesion classification, focused on melanoma diagnosis and staging from dermoscopic images.

The system is based on three binary classification tasks, trained independently using cross-validation, and combined at inference time into a sequential decision pipeline with majority voting.

---

## Problem Definition

Each dermoscopic image is ultimately classified into one of the following four clinical categories:

| Final Label | Description                        |
| ----------- | ---------------------------------- |
| 0           | Dysplastic Nevus                   |
| 1           | Melanoma in situ                   |
| 2           | Invasive Melanoma (Breslow < 1 mm) |
| 3           | Invasive Melanoma (Breslow ≥ 1 mm) |

To reflect the clinical decision process, classification is performed sequentially using three tasks:

1. Task 1: Dysplastic Nevus vs Melanoma
2. Task 2: Melanoma *in situ* vs Invasive Melanoma
3. Task 3: Breslow thickness < 1 mm vs ≥ 1 mm

---

## Model Architecture

* Backbone: DenseNet121 (ImageNet pretrained)
* Input size: 224 × 224 RGB images
* Output:
  * Binary softmax classifiers for Tasks 1–3  
* Training strategy:
  * 5-fold cross-validation per task
  * Class weighting to handle imbalance
  * Early stopping and best-model checkpointing

At inference time, predictions from the 5 folds are averaged (majority voting) for each task.

---


## Training

Training is performed independently for each task using cross-validation.

### Command

```bash
python src/train_val_test.py \
  --backbone densenet121 \
  --index <EXP_ID> \
  --task <TASK_ID>
```

### Arguments

| Argument     | Description                                       |
| ------------ | ------------------------------------------------- |
| `--backbone` | CNN backbone (`densenet121`, `vgg16`, `resnet50`) |
| `--index`    | Experiment identifier (used for folder naming)    |
| `--task`     | Task to train (1, 2, 3 or 4)                      |
| `--n_folds`  | Number of folds (default: 5)                      |

---

## Sequential Inference (Majority Voting)

The final sequential system combines the three tasks sequentially.

### Command

```bash
python src/test_sequential_classification_system_majority_voting.py \
  --exp1 <EXP_TASK1> \
  --exp2 <EXP_TASK2> \
  --exp3 <EXP_TASK3> \
  --threshold <THRESHOLD> \
  --external_test <0|1>
```

### Arguments

| Argument          | Description                                            |
| ----------------- | ------------------------------------------------------ |
| `--exp1`          | Experiment ID for Task 1 (Nevus vs Melanoma)           |
| `--exp2`          | Experiment ID for Task 2 (Mis vs Miv)                  |
| `--exp3`          | Experiment ID for Task 3 (Breslow)                     |
| `--threshold`     | Optional custom threshold for Task 1 (default: argmax) |
| `--external_test` | 0: internal test set, 1: external dataset              |

---

## Evaluation Metrics

The system reports:

* Confusion Matrix (normalized and raw)
* Balanced Accuracy
* Weighted AUC (OvR)
* Cohen’s Quadratic Kappa
* Recall
* F1-score
* Weighted Specificity

All results are automatically saved as CSV files and SVG figures.

---

## Notes

* CSV files must include at least: `path`, `label`
* Label encoding:

  * 0: Dysplastic Nevus
  * 1: Melanoma in situ
  * 2: Invasive melanoma (Breslow < 1 mm)
  * 3: Invasive melanoma (Breslow ≥ 1 mm)
* Images are loaded on-the-fly from disk

---

## License

This project is intended for research and academic use only.

---

## Contact

For questions or collaborations, please open an issue or contact the repository owner.
