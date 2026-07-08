# Alzheimer's Disease MRI Classification with CAM-Based Explainable AI

This repository contains training, visualization, and project-explainer code for
Alzheimer's disease stage classification from 2D MRI images using ResNet-18 and
CAM-based explainable AI methods.

The work focuses on two goals:

1. Classify MRI images into four dementia-stage categories.
2. Improve interpretability by showing which image regions influenced the model.

## Interactive Project Page

The repository includes a Streamlit page that explains the project idea,
pipeline, visual attention examples, evaluation approach, and key learning
outcomes.

Run it locally:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The app is an educational research demo only. It does not provide medical
diagnosis and should not be used for clinical decisions.

## Project Overview

The models classify MRI images into four classes:

- NonDemented
- VeryMildDemented
- MildDemented
- ModerateDemented

The training scripts use:

- ResNet-18 backbone
- Input size: 224 x 224
- Epochs: 80
- Batch size: 16
- Optimizer: Adam
- Learning rate: 0.001
- Train/validation split: 80/20

## Included Files

### Streamlit app

- `app.py` - project overview, pipeline explanation, visual explorer, evaluation notes, and learning summary
- `assets/presentation/` - selected images extracted from the project presentation for the app

### Training scripts

- `1_train_baseline_model.py` - baseline ResNet-18 classifier
- `2_train_gradcam_model.py` - Grad-CAM enhanced dual-branch model
- `3_train_scorecam_model.py` - ScoreCAM enhanced dual-branch model
- `4_train_layercam_model.py` - LayerCAM enhanced dual-branch model

### Heatmap generation scripts

- `generate_heatmaps_gradcam.py`
- `generate_heatmaps_scorecam.py`
- `generate_heatmaps_layercam.py`

### Documentation

- `NN_FinalReport.pdf` - final project report
- `NN_FinalPresentation.pptx` - final project presentation
- `README_HEATMAPS.txt` - heatmap-generation notes
- `requirements.txt` - Streamlit app dependencies

## Model Variants

### Baseline ResNet-18

Reference classifier using a pretrained ResNet-18 with a final layer adapted for
four dementia-stage classes.

### Grad-CAM Enhanced Model

Adds visual explanation based on class-discriminative gradients from the final
convolutional layer.

### ScoreCAM Enhanced Model

Uses perturbation-based class activation scoring. The project materials present
this as the strongest visual explanation direction.

### LayerCAM Enhanced Model

Uses layer-wise activation explanations, intended as a lower-overhead
interpretability variant.

## Expected Dataset Structure

The training scripts expect a class-wise dataset:

```text
AugmentedAlzheimerDataset/
|-- NonDemented/
|-- VeryMildDemented/
|-- MildDemented/
`-- ModerateDemented/
```

Supported image extensions:

- `.jpg`
- `.png`
- `.jpeg`

## Important Setup Note

The current training and heatmap scripts still contain hardcoded Windows paths
inside their `CONFIG` dictionaries, for example:

```python
"dataset_path": r"D:\cop\AugmentedAlzheimerDataset"
"output_path": r"D:\cop\AugmentedAlzheimerDataset\roi_scorecam_output\evaluation_results"
```

Before running training or heatmap generation, update those paths or refactor
them into command-line arguments/config files.

## Training Commands

```bash
python 1_train_baseline_model.py
python 2_train_gradcam_model.py
python 3_train_scorecam_model.py
python 4_train_layercam_model.py
```

## Heatmap Commands

```bash
python generate_heatmaps_gradcam.py
python generate_heatmaps_scorecam.py
python generate_heatmaps_layercam.py
```

## Evaluation Approach

The project is structured to evaluate both classification performance and
interpretability.

The training scripts track:

- training loss
- validation loss
- validation accuracy
- best validation accuracy
- training history JSON files
- training-curve images
- best model checkpoints

The heatmap scripts support qualitative review by generating class activation
maps for Grad-CAM, ScoreCAM, and LayerCAM. This helps a reviewer inspect whether
the model appears to focus on relevant MRI regions instead of unrelated image
artifacts.

The final report includes the following performance highlights:

| Metric | Reported value | Context |
| --- | --- | --- |
| Baseline final accuracy | 97.12% | ResNet-18 baseline accuracy curve |
| Grad-CAM improvement | +0.63 percentage points | Accuracy improvement over baseline |
| LayerCAM improvement | +0.76 percentage points | Accuracy improvement over baseline |
| ScoreCAM final accuracy | 98.15% in the accuracy section; 98.25% in the conclusion | Best-performing CAM variant |
| ScoreCAM macro-F1 | 0.9820 | Balanced class-wise performance |
| ScoreCAM precision / recall | 98.55% / 98.38% | Conclusion summary |
| ScoreCAM error rate | 1.72% | Conclusion summary |
| ScoreCAM overhead | 4.0% | Conclusion summary |

The report also discusses confusion matrices for the ResNet-18 baseline and the
Grad-CAM, LayerCAM, and ScoreCAM variants. The repository does not currently
include those confusion-matrix images or raw metric files as standalone assets,
so adding them would make the evaluation story easier to reproduce and verify.

## Recommended Validation Artifacts

- `results_summary.json` with the report-backed final metrics for each model
- confusion-matrix images for baseline, Grad-CAM, ScoreCAM, and LayerCAM variants
- per-class precision, recall, and F1 score tables
- training-curve images committed to an `assets/results/` folder
- representative heatmap outputs for each dementia-stage class

## Suggested Next Improvements

- Add `config.yaml` or CLI arguments for all paths.
- Save `results_summary.json` with final metrics.
- Add confusion matrix and classification report outputs.
- Add a small permitted sample-assets folder.
- Add checkpoint loading for a real inference demo.
- Refactor shared dataset/model utilities into reusable modules.
