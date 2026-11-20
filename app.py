
import os
import pandas as pd
import streamlit as st

from model_utils import (
    load_model,
    predict_toxicity,
    METRICS_PATH,
    DEVICE
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Toxic Comment Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 750;
        margin-bottom: 0;
    }

    .subtitle {
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 15px;
    }

    .detected {
        color: #ff5c5c;
        font-weight: 700;
    }

    .not-detected {
        color: #16a36a;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 15px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. PROJECT HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ Toxic Comment Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Based Multi-Label Text Classification using BiLSTM'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def get_model_artifacts():
    return load_model()


try:
    artifacts = get_model_artifacts()

except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# ============================================================
# 5. LOAD TEST METRICS
# ============================================================

@st.cache_data
def load_metrics():

    if not os.path.exists(METRICS_PATH):
        return pd.DataFrame()

    return pd.read_csv(METRICS_PATH)


metrics_df = load_metrics()

overall_metrics = (
    metrics_df[metrics_df["Category"] == "Overall"].copy()
    if not metrics_df.empty and "Category" in metrics_df.columns
    else pd.DataFrame()
)

label_metrics = (
    metrics_df[metrics_df["Category"] == "Per-label"].copy()
    if not metrics_df.empty and "Category" in metrics_df.columns
    else pd.DataFrame()
)


# ============================================================
# 6. CATEGORY DISPLAY NAMES
# ============================================================

display_names = {
    "toxic": "Toxic",
    "severe_toxic": "Severe Toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity Hate"
}


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ Project Menu")

    st.caption("Toxic Comment Detection")

    st.divider()

    st.subheader("Model Information")

    st.write("**Architecture:** BiLSTM")

    st.write("**Task:** Multi-label Classification")

    st.write("**Framework:** PyTorch")

    st.write("**Device:**", str(DEVICE).upper())

    st.write(
        "**Maximum Sequence Length:**",
        artifacts["max_length"]
    )

    st.write(
        "**Vocabulary Size:**",
        f'{len(artifacts["vocabulary"]):,}'
    )

    st.divider()

    st.subheader("Detection Categories")

    for label in artifacts["labels"]:
        st.write("•", display_names.get(label, label))

    st.divider()

    st.caption(
        "Model-generated predictions are estimates. "
        "Review results before making moderation decisions."
    )


# ============================================================
# 8. DASHBOARD TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Live Prediction",
    "📊 Model Performance",
    "📈 Category Analysis"
])


# ============================================================
# TAB 1: LIVE COMMENT PREDICTION
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">Analyze a Comment</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter any sentence below. The trained BiLSTM model "
        "will analyze it across all six toxicity categories."
    )

    with st.form("prediction_form"):

        comment = st.text_area(
            "Enter your comment",
            height=160,
            placeholder="Type or paste any sentence here..."
        )

        submitted = st.form_submit_button(
            "🔍 Analyze Comment",
            use_container_width=True,
            type="primary"
        )

    if submitted:

        if not comment.strip():

            st.warning(
                "Please enter a sentence before analyzing."
            )

        else:

            try:

                with st.spinner("Analyzing your comment..."):

                    overall_result, results = predict_toxicity(
                        comment,
                        artifacts
                    )

                st.divider()

                st.subheader("Prediction Result")

                if overall_result == "Potentially Toxic":

                    st.error(
                        "⚠️ Potentially Toxic Comment"
                    )

                    st.write(
                        "The model detected one or more toxicity categories."
                    )

                else:

                    st.success(
                        "No Toxicity Label Detected"
                    )

                    st.write(
                        "The model did not detect any category above "
                        "its selected threshold."
                    )

                # Category-wise prediction results

                st.subheader("Six-Category Analysis")

                for _, row in results.iterrows():

                    label = row["Toxicity Category"]

                    display_label = display_names.get(
                        label,
                        label
                    )

                    probability = float(row["Probability"])

                    threshold = float(row["Threshold"])

                    detected = row["Prediction"] == "Detected"

                    with st.container(border=True):

                        col1, col2 = st.columns([3, 1])

                        with col1:

                            st.markdown(
                                f"**{display_label}**"
                            )

                            st.progress(
                                max(0.0, min(1.0, probability))
                            )

                            st.caption(
                                f"Probability: {probability:.2%} "
                                f"| Detection threshold: {threshold:.2%}"
                            )

                        with col2:

                            if detected:

                                st.markdown(
                                    '<span class="detected">'
                                    'DETECTED</span>',
                                    unsafe_allow_html=True
                                )

                            else:

                                st.markdown(
                                    '<span class="not-detected">'
                                    'NOT DETECTED</span>',
                                    unsafe_allow_html=True
                                )

                st.caption(
                    "Each category uses its own saved threshold. "
                    "A category is detected when its predicted "
                    "probability meets or exceeds that threshold."
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )


# ============================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">Test Set Evaluation</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Evaluation results from the saved test_metrics.csv file."
    )

    if overall_metrics.empty:

        st.warning(
            "Overall evaluation metrics are unavailable."
        )

    else:

        metric_lookup = dict(
            zip(
                overall_metrics["Metric"],
                overall_metrics["Score"]
            )
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Micro F1-Score",
            f'{metric_lookup.get("Micro F1", 0):.4f}'
        )

        metric2.metric(
            "Macro F1-Score",
            f'{metric_lookup.get("Macro F1", 0):.4f}'
        )

        metric3.metric(
            "Subset Accuracy",
            f'{metric_lookup.get("Subset Accuracy", 0):.4f}'
        )

        metric4, metric5, metric6 = st.columns(3)

        metric4.metric(
            "Micro Precision",
            f'{metric_lookup.get("Micro Precision", 0):.4f}'
        )

        metric5.metric(
            "Micro Recall",
            f'{metric_lookup.get("Micro Recall", 0):.4f}'
        )

        metric6.metric(
            "Weighted F1-Score",
            f'{metric_lookup.get("Weighted F1", 0):.4f}'
        )

        st.divider()

        st.subheader("Complete Evaluation Metrics")

        st.dataframe(
            overall_metrics[["Metric", "Score"]].style.format(
                {"Score": "{:.4f}"}
            ),
            use_container_width=True,
            hide_index=True
        )

        if not label_metrics.empty:

            st.subheader("Per-Category Evaluation")

            display_metrics = label_metrics[
                [
                    "Label",
                    "Precision",
                    "Recall",
                    "F1-score",
                    "Average precision",
                    "Support",
                    "Threshold"
                ]
            ].copy()

            display_metrics["Label"] = (
                display_metrics["Label"].map(
                    lambda x: display_names.get(x, x)
                )
            )

            st.dataframe(
                display_metrics.style.format({
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-score": "{:.4f}",
                    "Average precision": "{:.4f}",
                    "Threshold": "{:.2f}",
                    "Support": "{:.0f}"
                }),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# TAB 3: CATEGORY ANALYSIS
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">Toxicity Category Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Compare model performance across all six toxicity categories."
    )

    if label_metrics.empty:

        st.warning(
            "Per-category metrics are unavailable."
        )

    else:

        chart_data = label_metrics[
            [
                "Label",
                "Precision",
                "Recall",
                "F1-score"
            ]
        ].copy()

        chart_data["Label"] = chart_data["Label"].map(
            lambda x: display_names.get(x, x)
        )

        chart_data = chart_data.set_index("Label")

        st.subheader("Precision, Recall and F1-Score")

        st.bar_chart(
            chart_data[
                ["Precision", "Recall", "F1-score"]
            ],
            y_label="Score",
            x_label="Toxicity Category"
        )

        st.divider()

        st.subheader("F1-Score Comparison")

        f1_data = chart_data[
            ["F1-score"]
        ].sort_values(
            "F1-score",
            ascending=False
        )

        st.bar_chart(
            f1_data,
            y_label="F1-Score",
            x_label="Toxicity Category"
        )

        st.caption(
            "All category-wise results are taken from "
            "the held-out test-set evaluation."
        )


# ============================================================
# 9. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <center>
        <b>Toxic Comment Detection System</b><br>
        Bidirectional LSTM | Multi-Label Classification<br>
        Deep Learning Project
    </center>
    """,
    unsafe_allow_html=True
)
