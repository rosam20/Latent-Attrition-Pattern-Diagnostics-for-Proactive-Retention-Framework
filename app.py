import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sqlite3
from datetime import datetime

# Optional: SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# For model comparison
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# ---------------------------
# Helpers
# ---------------------------
def try_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def encode_gender_series(s: pd.Series):
    s_clean = s.fillna("").astype(str).str.strip()
    lower = s_clean.str.lower()
    if set(lower.unique()) <= {"male", "female"} or ({"male","female"} & set(lower.unique())):
        return lower.map({"male": 0, "female": 1}).astype(float)
    out = pd.to_numeric(s_clean, errors="coerce")
    if out.isna().all():
        mapping = {v:i for i,v in enumerate(s_clean.unique())}
        return s_clean.map(mapping).astype(float)
    return out.fillna(0)

def ensure_required_columns(df, cols):
    return all(c in df.columns for c in cols)

def safe_predict(model, X_scaled):
    preds = model.predict(X_scaled)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_scaled)[:, 1]
        except Exception:
            probs = None
    return preds, probs

def cluster_and_plot(X_scaled, df_with_preds, k):
    try:
        kms = KMeans(n_clusters=k, random_state=42, n_init="auto")
        clusters = kms.fit_predict(X_scaled)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        plot_df = pd.DataFrame({
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "cluster": clusters
        })
        if "prediction" in df_with_preds.columns:
            plot_df["Prediction"] = df_with_preds["prediction"].values
        st.subheader("Customer Segmentation (KMeans + PCA)")
        fig, ax = plt.subplots()
        ax.scatter(plot_df["pc1"], plot_df["pc2"], c=plot_df["cluster"], alpha=0.8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"KMeans clusters (k={k})")
        st.pyplot(fig)
    except Exception:
        st.info("Clustering visualization unavailable.")

def basic_charts(df_with_preds):
    st.subheader("Churn Distribution")
    churn_count = (df_with_preds["prediction"] == "Churn").sum()
    stay_count  = (df_with_preds["prediction"] == "Stay").sum()
    fig, ax = plt.subplots()
    ax.pie([stay_count, churn_count], labels=["Stay", "Churn"], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    if "gender" in df_with_preds.columns:
        st.subheader("Churn % by Gender")
        tmp = df_with_preds.copy()
        if tmp["gender"].dtype != object:
            tmp["gender"] = tmp["gender"].map({0: "Male", 1: "Female"}).fillna(tmp["gender"])
        table = pd.crosstab(tmp["gender"], tmp["prediction"], normalize="index") * 100
        fig2, ax2 = plt.subplots()
        table.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Percent (%)")
        st.pyplot(fig2)

    for col in ["age", "tenure", "monthlycharge"]:
        if col in df_with_preds.columns:
            st.subheader(f"Distribution of {col}")
            fig3, ax3 = plt.subplots()
            df_with_preds[col].plot.hist(ax=ax3, bins=20)
            st.pyplot(fig3)

# ---------------------------
# Retention Strategy
# ---------------------------
def suggest_retention_strategy(row):
    if row["prediction"] == "Stay":
        return "No action needed"
    strategies = []
    if row.get("tenure",0) < 12:
        strategies.append("Offer welcome discounts / early loyalty rewards")
    if row.get("monthlycharge",0) > 70:
        strategies.append("Provide flexible billing or cost reduction options")
    if row.get("age",0) < 25:
        strategies.append("Engage with youth-oriented offers or bundles")
    if row.get("age",0) > 60:
        strategies.append("Provide senior-friendly plans with extra support")
    if not strategies:
        strategies.append("Personalized engagement via customer care")
    return "; ".join(strategies)

def highlight_strategy_card(row):
    st.markdown(
        f"""
        <div style="background-color:#ffe6e6;padding:12px;border-radius:10px;margin-bottom:10px;color:black;">
        <b>👤 Customer:</b> Age={row['age']}, Gender={row['gender']}, Tenure={row['tenure']} months, Charges=${row['monthlycharge']}<br>
        <b style="color:#cc0000;">⚠️ Retention Strategy:</b> <span style="color:#000000;">{row['Retention_Strategy']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------
# Confidence Band
# ---------------------------
def show_confidence_band(prob):
    if prob >= 80:
        color = "red"
        level = "High Risk ⚠️"
    elif prob >= 50:
        color = "orange"
        level = "Medium Risk ⚡"
    else:
        color = "green"
        level = "Low Risk ✅"
    st.markdown(
        f"""
        <div style="background-color:{color};padding:10px;border-radius:8px;color:white;text-align:center;">
        <b>Prediction Confidence:</b> {prob:.2f}% → {level}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------
# SQLite Prediction History
# ---------------------------
def init_db():
    conn = sqlite3.connect("prediction_history.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    age INTEGER,
    gender TEXT,
    tenure INTEGER,
    monthlycharge REAL,
    prediction TEXT,
    churn_prob REAL,
    retention_strategy TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_prediction_to_db(df: pd.DataFrame):
    conn = sqlite3.connect("prediction_history.db")

    df_to_save = df.copy()
    df_to_save["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ✅ Ensure columns exist (avoid KeyError)
    if "Churn_Prob" not in df_to_save.columns:
        df_to_save["Churn_Prob"] = None

    if "Retention_Strategy" not in df_to_save.columns:
        df_to_save["Retention_Strategy"] = None

    # ✅ Use correct column names (match your dataframe)
    allowed_cols = [
        "timestamp", "age", "gender", "tenure",
        "monthlycharge", "prediction",
        "Churn_Prob", "Retention_Strategy"
    ]

    df_to_save = df_to_save[allowed_cols]
    df_to_save.to_sql("history", conn, if_exists="append", index=False)

    conn.close()

def load_history():
    conn = sqlite3.connect("prediction_history.db")
    df = pd.read_sql("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

# ---------------------------
# Load persisted artifacts
# ---------------------------
st.set_page_config(page_title="Ultimate Churn Predictor", layout="wide")
st.title("Ultimate Customer Churn Prediction Suite")

model = try_load("model.pkl")
scaler = try_load("scaler.pkl")
le_gender = try_load("le_gender.pkl")

if model is None or scaler is None:
    st.error("Missing `model.pkl` or `scaler.pkl`.")
    st.stop()

FEATURES = ["age", "gender", "tenure", "monthlycharge"]
has_proba = hasattr(model, "predict_proba")

# ---------------------------
# Tabs (Only the essential ones)
# ---------------------------
tab_dash, tab_single, tab_batch, tab_history, tab_impact = st.tabs(
["📊 Dashboard", "👤 Single Prediction", "📂 Batch Prediction", "📜 History", "💰 Business Impact"]
)

# ===========================
# Dashboard Tab
# ===========================
with tab_dash:
    st.subheader("Upload a dataset to visualize churn and segment customers")
    uploaded_dash = st.file_uploader("Upload CSV for dashboard", type=["csv"], key="dash_up")
    k = st.slider("Number of clusters (KMeans)", min_value=2, max_value=8, value=3, step=1)

    if uploaded_dash is not None:
        df = pd.read_csv(uploaded_dash)
        df.columns = df.columns.str.lower()
        if "monthlycharges" in df.columns:
            df = df.rename(columns={"monthlycharges": "monthlycharge"})
        if ensure_required_columns(df, FEATURES):
            X_df = df[FEATURES].copy()
            if le_gender is not None:
                try:
                    X_df["gender"] = le_gender.transform(X_df["gender"].astype(str))
                except Exception:
                    X_df["gender"] = encode_gender_series(X_df["gender"])
            else:
                X_df["gender"] = encode_gender_series(X_df["gender"])
            for c in FEATURES:
                X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0)
            X_scaled = scaler.transform(X_df[FEATURES])
            preds, probs = safe_predict(model, X_scaled)
            df["prediction"] = np.where(preds == 1, "Churn", "Stay")
            if probs is not None:
                df["Churn_Prob"] = (probs * 100).round(2)
                df["Retention_Strategy"] = df.apply(suggest_retention_strategy, axis=1)

            churn_count = (df["prediction"] == "Churn").sum()
            stay_count  = (df["prediction"] == "Stay").sum()
            total = len(df)
            st.metric("Total customers", total)
            st.metric("Likely to churn", churn_count)
            st.metric("Likely to stay", stay_count)

            basic_charts(df)
            cluster_and_plot(X_scaled, df, k)

            st.subheader("Highlighted Retention Strategies")
            churners = df[df["prediction"] == "Churn"].head(5)
            if churners.empty:
                st.success("No churn customers detected 🎉")
            else:
                for _, row in churners.iterrows():
                    highlight_strategy_card(row)

            st.subheader("Preview with predictions (full data)")
            st.dataframe(df.head(20))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv", key="dash_csv")

            # Save dashboard predictions to history
            save_prediction_to_db(df)

# ===========================
# Single Prediction Tab
# ===========================
with tab_single:
    st.subheader("Enter one customer's details")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=110, value=30)
    with col2:
        gender_text = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=1000, value=12)
    with col4:
        monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=100000.0, value=50.0)

    if st.button("Predict"):
        X_one = pd.DataFrame([{
            "age": age,
            "gender": gender_text,
            "tenure": tenure,
            "monthlycharge": monthly
        }])
        if le_gender is not None:
            try:
                X_one["gender"] = le_gender.transform(X_one["gender"].astype(str))
            except Exception:
                X_one["gender"] = encode_gender_series(X_one["gender"])
        else:
            X_one["gender"] = encode_gender_series(X_one["gender"])
        for c in FEATURES:
            X_one[c] = pd.to_numeric(X_one[c], errors="coerce").fillna(0)

        X_scaled_one = scaler.transform(X_one[FEATURES])
        pred, prob = safe_predict(model, X_scaled_one)
        label = "Churn" if pred[0] == 1 else "Stay"

        X_one["prediction"] = label
        if prob is not None:
            X_one["Churn_Prob"] = prob[0]*100
            X_one["Retention_Strategy"] = suggest_retention_strategy({
                "prediction": label, "age": age, "tenure": tenure, "monthlycharge": monthly
            })

        # Save single prediction to history
        save_prediction_to_db(X_one)

        st.subheader("Result")
        if label == "Churn":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        if prob is not None:
            st.info(f"Probability of churn: {prob[0]*100:.2f}%")
            show_confidence_band(prob[0]*100)

        strategy = X_one.loc[0, "Retention_Strategy"]
        st.markdown(
            f"""
            <div style="background-color:#fff0cc;padding:12px;border-radius:10px;margin-top:10px;color:black;">
            <b style="color:#e65c00;">💡 Suggested Retention Strategy:</b>
            <span style="color:#000000;">{strategy}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ===========================
# Batch Prediction Tab
# ===========================
with tab_batch:
    st.subheader("Batch prediction — upload CSV")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="batch_up")
    if uploaded is not None:
        dfb = pd.read_csv(uploaded)
        dfb.columns = dfb.columns.str.lower()
        if "monthlycharges" in dfb.columns:
            dfb = dfb.rename(columns={"monthlycharges": "monthlycharge"})
        if ensure_required_columns(dfb, FEATURES):
            Xb = dfb[FEATURES].copy()
            if le_gender is not None:
                try:
                    Xb["gender"] = le_gender.transform(Xb["gender"].astype(str))
                except Exception:
                    Xb["gender"] = encode_gender_series(Xb["gender"])
            else:
                Xb["gender"] = encode_gender_series(Xb["gender"])
            for c in FEATURES:
                Xb[c] = pd.to_numeric(Xb[c], errors="coerce").fillna(0)
            Xb_scaled = scaler.transform(Xb[FEATURES])
            preds, probs = safe_predict(model, Xb_scaled)
            dfb["prediction"] = np.where(preds == 1, "Churn", "Stay")
            if probs is not None:
                dfb["Churn_Prob"] = (probs * 100).round(2)
                dfb["Retention_Strategy"] = dfb.apply(suggest_retention_strategy, axis=1)

            churn_count = (dfb["prediction"] == "Churn").sum()
            stay_count  = (dfb["prediction"] == "Stay").sum()
            total = len(dfb)
            st.metric("Total", total)
            st.metric("Churn", churn_count)
            st.metric("Stay", stay_count)

            st.subheader("Highlighted Retention Strategies")
            churners_b = dfb[dfb["prediction"] == "Churn"].head(5)
            if churners_b.empty:
                st.success("No churn customers detected 🎉")
            else:
                for _, row in churners_b.iterrows():
                    highlight_strategy_card(row)

            st.subheader("Predictions (full data)")
            st.dataframe(dfb.head(20))

            csv = dfb.to_csv(index=False).encode("utf-8")
            st.download_button("Download batch predictions (CSV)", data=csv, file_name="batch_predictions.csv", mime="text/csv", key="batch_csv")

            # Save batch predictions to history
            save_prediction_to_db(dfb)

# History Tab
with tab_history:
    st.subheader("Prediction History")
    hist_df = load_history()
    st.dataframe(hist_df)

# ===========================
# Business Impact Tab
# ===========================
with tab_impact:
    st.subheader("💰 Business Impact Simulation")
    uploaded_impact = st.file_uploader("Upload CSV for business impact", type=["csv"], key="impact_up")

    if uploaded_impact is not None:
        df_impact = pd.read_csv(uploaded_impact)
        df_impact.columns = df_impact.columns.str.lower()
        if "monthlycharges" in df_impact.columns:
            df_impact = df_impact.rename(columns={"monthlycharges": "monthlycharge"})
        required_cols = ["age", "gender", "tenure", "monthlycharge"]
        if not ensure_required_columns(df_impact, required_cols):
            st.error(f"CSV must include these columns: {required_cols}")
        else:
            X_impact = df_impact[required_cols].copy()
            if le_gender is not None:
                try:
                    X_impact["gender"] = le_gender.transform(X_impact["gender"].astype(str))
                except Exception:
                    X_impact["gender"] = encode_gender_series(X_impact["gender"])
            else:
                X_impact["gender"] = encode_gender_series(X_impact["gender"])

            for c in required_cols:
                X_impact[c] = pd.to_numeric(X_impact[c], errors="coerce").fillna(0)

            # Predict churn if Prediction column not present
            if "prediction" not in df_impact.columns:
                X_scaled_impact = scaler.transform(X_impact[required_cols])
                preds, probs = safe_predict(model, X_scaled_impact)
                df_impact["prediction"] = np.where(preds == 1, "Churn", "Stay")
                if probs is not None:
                    df_impact["Churn_Prob"] = (probs * 100).round(2)

            # Retention strategy
            df_impact["Retention_Strategy"] = df_impact.apply(suggest_retention_strategy, axis=1)

            # Revenue Loss Simulation
            avg_revenue_per_customer = df_impact["monthlycharge"].mean() * 12
            churners_count = (df_impact["prediction"] == "Churn").sum()
            predicted_loss = churners_count * avg_revenue_per_customer * 0.2
            savings = predicted_loss * 0.5

            # Metrics
            st.metric("Predicted Revenue Loss (20% churn)", f"${predicted_loss:,.2f}")
            st.metric("Potential Savings from Retention", f"${savings:,.2f}")
            st.metric("Total Customers", len(df_impact))
            st.metric("Predicted Churn Customers", churners_count)

            # Charts
            st.subheader("Churn by Gender")
            tmp = df_impact.copy()
            if tmp["gender"].dtype != object:
                tmp["gender"] = tmp["gender"].map({0: "Male", 1: "Female"}).fillna(tmp["gender"])
            chart_df = pd.crosstab(tmp["gender"], tmp["prediction"])
            st.bar_chart(chart_df)

            st.subheader("Monthly Charges Distribution for Churned Customers")
            churned_customers = df_impact[df_impact["prediction"] == "Churn"]
            if not churned_customers.empty:
                fig, ax = plt.subplots()
                ax.hist(churned_customers["monthlycharge"], bins=15, color="#e65c00", alpha=0.7)
                ax.set_xlabel("monthlycharge")
                ax.set_ylabel("Count")
                ax.set_title("monthlycharge distribution (Churned Customers)")
                st.pyplot(fig)
            else:
                st.info("No churn customers detected for chart.")

            st.subheader("Preview with predictions")
            st.dataframe(df_impact.head(20))

            csv_impact = df_impact.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions (CSV)",
                data=csv_impact,
                file_name="business_impact_predictions.csv",
                mime="text/csv"
            )
