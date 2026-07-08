from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
ASSET_DIR = ROOT / "assets" / "presentation"

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

overview_tab, pipeline_tab, explorer_tab, learning_tab, gaps_tab = st.tabs(
    ["Overview", "Pipeline", "Visual Explorer", "Learning", "Next Steps"]
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
"""
    )

    left, right = st.columns(2)
    with left:
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
    with right:
        st.markdown(
            """
#### Portfolio Value

- Shows applied computer vision
- Connects ML performance with explainability
- Uses medical-imaging constraints responsibly
- Creates a visual story that hiring managers can understand quickly
"""
        )

with gaps_tab:
    st.subheader("Next Steps Before a Full Live Demo")
    st.markdown(
        """
The current repository is strongest as an explanatory project page. To turn it
into a live inference demo, the project needs a few missing assets and cleanup
items.
"""
    )

    gaps = pd.DataFrame(
        [
            {
                "Gap": "Model checkpoint",
                "Why it matters": "A trained `.pth` file is required for real predictions.",
                "Fix": "Add a small demo checkpoint or document where to download it.",
            },
            {
                "Gap": "Sample test set",
                "Why it matters": "The app needs safe example MRI images to run without private data.",
                "Fix": "Add a small permitted sample set or precomputed demo outputs.",
            },
            {
                "Gap": "Configurable paths",
                "Why it matters": "Hardcoded `D:\\cop\\...` paths block reproducibility.",
                "Fix": "Move dataset/model/output paths into CLI args or config.",
            },
            {
                "Gap": "Evaluation artifacts",
                "Why it matters": "Reviewers need confusion matrices and per-class metrics.",
                "Fix": "Save metric JSON, confusion matrix, and sample heatmaps.",
            },
            {
                "Gap": "Medical validation",
                "Why it matters": "Interpretability is not the same as clinical usefulness.",
                "Fix": "Frame as research/education and avoid diagnostic claims.",
            },
        ]
    )
    st.dataframe(gaps, hide_index=True, use_container_width=True)
