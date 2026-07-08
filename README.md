# Alzheimer's Disease MRI Classification with CAM-Based Explainable AI

This repository contains training, visualization, and project-explainer code for
Alzheimer's disease stage classification from 2D MRI images using ResNet-18 and
CAM-based explainable AI methods.

The work focuses on two goals:

1. Classify MRI images into four dementia-stage categories.
2. Improve interpretability by showing which image regions influenced the model.

## Interactive Project Page

The repository includes a Streamlit page that explains the project idea,
pipeline, visual attention examples, and the remaining gaps before a full live
inference demo.

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

- `app.py` - project overview, pipeline explanation, visual explorer, learning notes, and next-step gaps
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
├── NonDemented/
├── VeryMildDemented/
├── MildDemented/
└── ModerateDemented/
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

## Current Gaps

To make this a full live model demo, the repo still needs:

- trained checkpoint files or documented download links
- safe sample MRI images for demo inference
- confusion matrices and per-class metrics
- configurable dataset/model/output paths
- stronger reproducibility controls
- explicit medical/research-only safety framing

## Suggested Next Improvements

- Add `config.yaml` or CLI arguments for all paths.
- Save `results_summary.json` with final metrics.
- Add confusion matrix and classification report outputs.
- Add a small permitted sample-assets folder.
- Add checkpoint loading for a real inference demo.
- Refactor shared dataset/model utilities into reusable modules.
