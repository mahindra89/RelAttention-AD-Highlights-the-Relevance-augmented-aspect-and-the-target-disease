from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
ASSET_DIR = ROOT / "assets" / "presentation"
REPORT_PATH = ROOT / "NN_FinalReport.pdf"
PRESENTATION_PATH = ROOT / "NN_FinalPresentation.pptx"

CLASS_IMAGES = {
    "Mild Dementia": ASSET_DIR / "slide_04_image_01.jpg",
    "Moderate Dementia": ASSET_DIR / "slide_04_image_02.jpg",
    "Very Mild Dementia": ASSET_DIR / "slide_04_image_03.jpg",
    "Non Demented": ASSET_DIR / "slide_04_image_04.jpg",
}

ATTENTION_IMAGES = {
    "Mild Dementia - highlighted region": ASSET_DIR / "slide_11_image_01.jpg",
    "Moderate Dementia - highlighted region": ASSET_DIR / "slide_11_image_02.jpg",
    "Very Mild Dementia - highlighted region": ASSET_DIR / "slide_12_image_01.jpg",
    "Non Demented - highlighted region": ASSET_DIR / "slide_12_image_02.jpg",
    "Additional attention example": ASSET_DIR / "slide_12_image_03.jpg",
}

MODEL_TABLE = pd.DataFrame(
    [
        {
            "Variant": "Baseline ResNet-18",
            "Role": "Reference classifier",
            "Explainability": "None by default",
            "Portfolio takeaway": "Fast baseline for 4-stage MRI classification",
        },
        {
            "Variant": "Grad-CAM enhanced",
            "Role": "Gradient-based visual explanation",
            "Explainability": "Highlights class-discriminative regions",
            "Portfolio takeaway": "Makes the CNN decision process easier to inspect",
        },
        {
            "Variant": "ScoreCAM enhanced",
            "Role": "Perturbation-based attention scoring",
            "Explainability": "Most stable saliency-map direction in the project notes",
            "Portfolio takeaway": "Best candidate for the public visual demo",
        },
        {
            "Variant": "LayerCAM enhanced",
            "Role": "Layer-wise activation explanation",
            "Explainability": "Lower-overhead attention visualization",
            "Portfolio takeaway": "Useful speed/interpretability comparison point",
        },
    ]
)

EVALUATION_TABLE = pd.DataFrame(
    [
        {
            "Evidence area": "Classification quality",
            "What the code tracks": "Training loss, validation loss, and validation accuracy",
            "Current repo evidence": "Training scripts save history JSON and best checkpoints when run",
            "Why it matters": "Shows whether the model learns beyond a visual demo",
        },
        {
            "Evidence area": "Model comparison",
            "What the code tracks": "Baseline, Grad-CAM, ScoreCAM, and LayerCAM variants",
            "Current repo evidence": "Separate scripts use the same dataset split and backbone family",
            "Why it matters": "Makes the project a controlled experiment rather than one model run",
        },
        {
            "Evidence area": "Explainability",
            "What the code tracks": "CAM heatmaps for sample images by class",
            "Current repo evidence": "Presentation examples and heatmap-generation scripts",
            "Why it matters": "Helps reviewers inspect whether predictions rely on relevant regions",
        },
        {
            "Evidence area": "Clinical responsibility",
            "What the code tracks": "Research-only framing and visual review",
            "Current repo evidence": "Public app disclaimer and interpretability notes",
            "Why it matters": "Avoids overstating a medical-imaging prototype",
        },
    ]
)

VALIDATION_ARTIFACTS = pd.DataFrame(
    [
        {
            "Artifact": "Training history JSON",
            "Produced by": "Each training script",
            "Status": "Supported by code",
            "Use": "Plot accuracy/loss and compare training stability",
        },
        {
            "Artifact": "Best model checkpoint",
            "Produced by": "Each training script",
            "Status": "Expected after training",
            "Use": "Reload the strongest model for heatmap generation",
        },
        {
            "Artifact": "Training curves",
            "Produced by": "Training scripts",
            "Status": "Supported by code",
            "Use": "Show convergence and overfitting behavior",
        },
        {
            "Artifact": "Confusion matrix",
            "Produced by": "Not yet included",
            "Status": "Recommended next metric",
            "Use": "Reveal which dementia-stage classes are confused",
        },
        {
            "Artifact": "Per-class precision/recall/F1",
            "Produced by": "Not yet included",
            "Status": "Recommended next metric",
            "Use": "Evaluate class imbalance and minority-class behavior",
        },
    ]
)

REPORT_METRICS = pd.DataFrame(
    [
        {
            "Metric from report": "Baseline final accuracy",
            "Reported value": "97.12%",
            "Context": "ResNet-18 baseline accuracy curve",
        },
        {
            "Metric from report": "Grad-CAM improvement",
            "Reported value": "+0.63 percentage points",
            "Context": "Accuracy improvement over baseline",
        },
        {
            "Metric from report": "LayerCAM improvement",
            "Reported value": "+0.76 percentage points",
            "Context": "Accuracy improvement over baseline",
        },
        {
            "Metric from report": "ScoreCAM final accuracy",
            "Reported value": "98.15% in accuracy section; 98.25% in conclusion",
            "Context": "Best-performing CAM variant",
        },
        {
            "Metric from report": "ScoreCAM macro-F1",
            "Reported value": "0.9820",
            "Context": "Balanced class-wise performance",
        },
        {
            "Metric from report": "ScoreCAM precision / recall",
            "Reported value": "98.55% / 98.38%",
            "Context": "Conclusion summary",
        },
        {
            "Metric from report": "ScoreCAM error rate",
            "Reported value": "1.72%",
            "Context": "Conclusion summary",
        },
        {
            "Metric from report": "ScoreCAM overhead",
            "Reported value": "4.0%",
            "Context": "Conclusion summary",
        },
    ]
)


st.set_page_config(page_title="Alzheimer MRI Explainability", layout="wide")

st.title("Alzheimer MRI Explainability")
st.caption(
    "A project overview for 2D MRI dementia-stage classification with CNN-based "
    "attention visualizations."
)

st.warning(
    "Educational research demo only. This page does not provide medical diagnosis "
    "and should not be used for clinical decisions."
)

download_left, download_right = st.columns(2)
with download_left:
    if REPORT_PATH.exists():
        st.download_button(
            "Download report",
            data=REPORT_PATH.read_bytes(),
            file_name="NN_FinalReport.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
with download_right:
    if PRESENTATION_PATH.exists():
        st.download_button(
            "Download presentation",
            data=PRESENTATION_PATH.read_bytes(),
            file_name="NN_FinalPresentation.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True,
        )

overview_tab, pipeline_tab, explorer_tab, evaluation_tab, learning_tab = st.tabs(
    ["Overview", "Pipeline", "Visual Explorer", "Evaluation", "Learning"]
)

with overview_tab:
    st.subheader("Project Idea")
    st.markdown(
        """
This project studies how deep learning can classify 2D brain MRI slices into
four dementia-stage categories while also showing *where* the model is looking.

The original inspiration is Rel-SA: a relevance-augmented attention approach
that guides model focus toward clinically meaningful brain regions. In this
implementation, the idea is adapted into a practical CNN-based workflow using
ResNet-18 and CAM-style explainability methods such as Grad-CAM, ScoreCAM, and
LayerCAM.
"""
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Classes", "4")
    col2.metric("Backbone", "ResNet-18")
    col3.metric("Input", "224 x 224")
    col4.metric("Focus", "Explainability")

    st.subheader("Classification Targets")
    image_cols = st.columns(4)
    for col, (label, path) in zip(image_cols, CLASS_IMAGES.items()):
        with col:
            if path.exists():
                st.image(str(path), caption=label, use_container_width=True)
            else:
                st.info(label)

    st.subheader("Model Variants")
    st.dataframe(MODEL_TABLE, use_container_width=True)

with pipeline_tab:
    st.subheader("End-to-End Workflow")
    st.markdown(
        """
The pipeline starts with MRI preprocessing, trains a baseline image classifier,
then adds attention-based visual explanations to compare interpretability and
model behavior.
"""
    )

    preprocessing = ASSET_DIR / "slide_05_image_01.png"
    if preprocessing.exists():
        st.image(
            str(preprocessing),
            caption="Preprocessing flow from the project presentation",
            use_container_width=True,
        )

    steps = pd.DataFrame(
        [
            {"Step": "1. Input", "Description": "Load 2D MRI images organized by dementia-stage class."},
            {"Step": "2. Preprocess", "Description": "Resize, normalize, augment, and prepare tensors for ResNet-18."},
            {"Step": "3. Train", "Description": "Train baseline and CAM-enhanced model variants."},
            {"Step": "4. Explain", "Description": "Generate attention maps that highlight influential image regions."},
            {"Step": "5. Compare", "Description": "Review accuracy, interpretability, and model behavior across variants."},
        ]
    )
    st.dataframe(steps, hide_index=True, use_container_width=True)

with explorer_tab:
    st.subheader("Attention Visualization Explorer")
    st.markdown(
        """
Use the selector to review example visualizations from the project presentation.
The highlighted regions represent areas the model explanation emphasized for a
given MRI example.
"""
    )

    selected = st.selectbox("Example", list(ATTENTION_IMAGES.keys()))
    selected_path = ATTENTION_IMAGES[selected]

    left, right = st.columns([1, 1])
    with left:
        if selected_path.exists():
            st.image(str(selected_path), caption=selected, use_container_width=True)
        else:
            st.warning("Example image is missing from assets.")

    with right:
        st.markdown(
            """
#### How to read this view

- Dark/low-emphasis regions are less influential in the explanation.
- Brighter highlighted regions are treated as more relevant to the model output.
- The goal is not only to classify the image, but to make the model's focus
  inspectable.
- For a medical ML system, interpretability should be validated with domain
  experts before any real-world use.
"""
        )

        st.info(
            "This page uses presentation assets and project summaries. Live model "
            "inference requires trained checkpoints, which are not currently included."
        )

with evaluation_tab:
    st.subheader("Evaluation Story")
    st.markdown(
        """
This project is evaluated from two angles: predictive performance and
explainability. The training scripts are designed to compare validation accuracy
across a baseline ResNet-18 model and three CAM-enhanced variants. The visual
explorer then adds qualitative evidence by showing which MRI regions the
explanation methods emphasize.

The final report includes metric summaries for the baseline and CAM-enhanced
models. The repository does not currently include the raw metric files or
confusion-matrix images as standalone assets, so this page separates
report-backed results from artifacts that should still be added for direct
reproducibility.
"""
    )

    st.subheader("Report-Backed Metrics")
    st.dataframe(REPORT_METRICS, hide_index=True, use_container_width=True)
    st.caption(
        "Note: the report describes ScoreCAM as 98.15% in the accuracy/key-metrics "
        "discussion and 98.25% in the conclusion. Both values are shown rather than "
        "silently choosing one."
    )

    st.dataframe(EVALUATION_TABLE, hide_index=True, use_container_width=True)

    st.subheader("Validation Artifacts")
    st.dataframe(VALIDATION_ARTIFACTS, hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
#### What is already strong

- Clear baseline-to-enhanced model comparison
- Consistent ResNet-18 backbone across variants
- Reported accuracy, macro-F1, precision, recall, and error-rate metrics
- CAM-based heatmap generation scripts
- Visual examples for multiple dementia-stage classes
"""
        )
    with right:
        st.markdown(
            """
#### What would make it stronger

- Add the report's confusion-matrix images as standalone assets
- Add per-class precision, recall, and F1 tables
- Add a small machine-readable `results_summary.json`
- Add training-curve images to the Streamlit page
"""
        )

with learning_tab:
    st.subheader("What This Project Demonstrates")
    st.markdown(
        """
The central learning goal is to connect classification performance with
interpretability. A standard CNN can predict a class, but a medical-imaging
workflow also needs to show why that prediction may be reasonable.

CAM methods provide a bridge between model output and visual explanation by
projecting class-relevant activations back onto the image. This allows a
reviewer to inspect whether the model appears to focus on meaningful anatomical
regions rather than image artifacts or background.

Overall, the project demonstrates applied computer vision in a high-impact
domain while treating medical-imaging constraints responsibly. It presents the
model as more than an accuracy score: the visual explanations create a clear
story about how performance, interpretability, and practical review can work
together in an AI-assisted imaging workflow.
"""
    )

    st.markdown(
        """
#### Technical Concepts

- Transfer learning with ResNet-18
- 4-class MRI image classification
- Grad-CAM, ScoreCAM, and LayerCAM
- Attention-style feature visualization
- Model comparison through accuracy and interpretability
"""
    )
