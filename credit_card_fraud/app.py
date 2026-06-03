import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb_lib
import matplotlib.pyplot as plt
import tensorflow as tf

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Load all models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    lr  = joblib.load("saved_models/lr_model.pkl")
    rf  = joblib.load("saved_models/rf_model.pkl")
    xgb = xgb_lib.XGBClassifier()
    xgb.load_model("saved_models/xgb_model.json")
    nn  = tf.keras.models.load_model("saved_models/nn_model.keras")
    scaler = joblib.load("saved_models/scaler.pkl")
    return lr, rf, xgb, nn, scaler

lr_model, rf_model, xgb_model, nn_model, scaler = load_models()

MODEL_MAP = {
    "Logistic Regression": lr_model,
    "Random Forest":       rf_model,
    "XGBoost":             xgb_model,
    "Neural Network":      nn_model,
}

# ── Helper: preprocess input ──────────────────────────────────
def preprocess(df: pd.DataFrame) -> np.ndarray:
    """Scale Time & Amount columns, leave V1–V28 as-is."""
    df = df.copy()
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    # Column order must match training
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    return df[cols].values

# ── Helper: predict with any model ───────────────────────────
def predict(model, X: np.ndarray):
    name = type(model).__name__

    if "keras" in name.lower() or hasattr(model, "predict_on_batch"):
        # Neural Network — returns probability directly
        prob = model.predict(X, verbose=0).flatten()
        pred = (prob >= 0.5).astype(int)
    else:
        pred = model.predict(X)
        prob = model.predict_proba(X)[:, 1]

    return pred, prob

# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════
st.title("🔍 Credit Card Fraud Detector")
st.markdown("Predict fraudulent transactions using four trained ML models.")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox(
        "Choose model",
        list(MODEL_MAP.keys())
    )
    threshold = st.slider(
        "Decision threshold",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="Lower = more sensitive to fraud (more false positives)"
    )
    st.divider()
    st.markdown("**Models available:**")
    for m in MODEL_MAP:
        icon = "✅" if m == selected_model else "⬜"
        st.markdown(f"{icon} {m}")

model = MODEL_MAP[selected_model]

# ── Input mode ────────────────────────────────────────────────
mode = st.radio("Input mode", ["Manual entry", "Upload CSV"], horizontal=True)

# ─────────────────────────────────────────────────────────────
# MANUAL ENTRY
# ─────────────────────────────────────────────────────────────
if mode == "Manual entry":
    st.subheader("Transaction Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        time   = st.number_input("Time (seconds elapsed)", value=0.0)
        amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
        v1  = st.number_input("V1",  value=0.0, format="%.4f")
        v2  = st.number_input("V2",  value=0.0, format="%.4f")
        v3  = st.number_input("V3",  value=0.0, format="%.4f")
        v4  = st.number_input("V4",  value=0.0, format="%.4f")
        v5  = st.number_input("V5",  value=0.0, format="%.4f")
        v6  = st.number_input("V6",  value=0.0, format="%.4f")
        v7  = st.number_input("V7",  value=0.0, format="%.4f")
        v8  = st.number_input("V8",  value=0.0, format="%.4f")
        v9  = st.number_input("V9",  value=0.0, format="%.4f")

    with col2:
        v10 = st.number_input("V10", value=0.0, format="%.4f")
        v11 = st.number_input("V11", value=0.0, format="%.4f")
        v12 = st.number_input("V12", value=0.0, format="%.4f")
        v13 = st.number_input("V13", value=0.0, format="%.4f")
        v14 = st.number_input("V14", value=0.0, format="%.4f")
        v15 = st.number_input("V15", value=0.0, format="%.4f")
        v16 = st.number_input("V16", value=0.0, format="%.4f")
        v17 = st.number_input("V17", value=0.0, format="%.4f")
        v18 = st.number_input("V18", value=0.0, format="%.4f")
        v19 = st.number_input("V19", value=0.0, format="%.4f")

    with col3:
        v20 = st.number_input("V20", value=0.0, format="%.4f")
        v21 = st.number_input("V21", value=0.0, format="%.4f")
        v22 = st.number_input("V22", value=0.0, format="%.4f")
        v23 = st.number_input("V23", value=0.0, format="%.4f")
        v24 = st.number_input("V24", value=0.0, format="%.4f")
        v25 = st.number_input("V25", value=0.0, format="%.4f")
        v26 = st.number_input("V26", value=0.0, format="%.4f")
        v27 = st.number_input("V27", value=0.0, format="%.4f")
        v28 = st.number_input("V28", value=0.0, format="%.4f")

    if st.button("🔎 Predict", type="primary"):
        row = pd.DataFrame([[time, v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
                              v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
                              v21,v22,v23,v24,v25,v26,v27,v28,amount]],
                           columns=["Time"]+[f"V{i}" for i in range(1,29)]+["Amount"])

        X = preprocess(row)
        pred, prob = predict(model, X)
        fraud_prob = prob[0]
        is_fraud   = fraud_prob >= threshold

        st.divider()
        r1, r2, r3 = st.columns(3)
        with r1:
            if is_fraud:
                st.error("🚨 FRAUDULENT transaction")
            else:
                st.success("✅ LEGITIMATE transaction")
        with r2:
            st.metric("Fraud Probability", f"{fraud_prob:.2%}")
            st.progress(float(fraud_prob))
        with r3:
            st.metric("Model used", selected_model)
            st.metric("Threshold", f"{threshold:.0%}")

# ─────────────────────────────────────────────────────────────
# CSV UPLOAD — BATCH PREDICTION
# ─────────────────────────────────────────────────────────────
else:
    st.subheader("Batch Prediction via CSV")
    st.info("CSV must have columns: **Time, V1–V28, Amount** (Class column optional)")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        true_labels = df["Class"].values if "Class" in df.columns else None

        st.write("**Preview (first 5 rows):**")
        st.dataframe(df.head(), use_container_width=True)

        X = preprocess(df.drop(columns=["Class"], errors="ignore"))
        preds, probs = predict(model, X)
        is_fraud = (probs >= threshold).astype(int)

        # Results dataframe
        results = df.copy()
        results["Fraud_Probability"] = probs.round(4)
        results["Prediction"]        = is_fraud
        results["Result"]            = np.where(is_fraud, "🚨 Fraud", "✅ Legit")

        st.divider()
        st.subheader(f"Results — {selected_model}")

        fraud_n = is_fraud.sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Transactions", len(df))
        m2.metric("Flagged as Fraud",   int(fraud_n))
        m3.metric("Legitimate",         int(len(df) - fraud_n))
        m4.metric("Fraud Rate",         f"{fraud_n/len(df):.2%}")

        # ── Model comparison (if true labels exist) ──────────
        if true_labels is not None:
            st.subheader("📊 All Models Comparison")
            from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

            comparison = []
            for mname, mdl in MODEL_MAP.items():
                p, pb = predict(mdl, X)
                flagged = (pb >= threshold).astype(int)
                comparison.append({
                    "Model":     mname,
                    "AUC-ROC":   f"{roc_auc_score(true_labels, pb):.4f}",
                    "F1 Score":  f"{f1_score(true_labels, flagged):.4f}",
                    "Precision": f"{precision_score(true_labels, flagged):.4f}",
                    "Recall":    f"{recall_score(true_labels, flagged):.4f}",
                })
            st.dataframe(pd.DataFrame(comparison), use_container_width=True)

        # ── Probability distribution chart ───────────────────
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(probs[is_fraud == 0], bins=40, alpha=0.7, label="Legit",  color="#2ecc71")
        ax.hist(probs[is_fraud == 1], bins=40, alpha=0.7, label="Fraud",  color="#e74c3c")
        ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold ({threshold})")
        ax.set_xlabel("Fraud Probability")
        ax.set_ylabel("Count")
        ax.set_title(f"Probability Distribution — {selected_model}")
        ax.legend()
        st.pyplot(fig)

        # ── Full results table ────────────────────────────────
        st.dataframe(results[["Time","Amount","Fraud_Probability","Result"]],
                     use_container_width=True)

        csv_out = results.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results", csv_out,
                           f"fraud_predictions_{selected_model}.csv", "text/csv")