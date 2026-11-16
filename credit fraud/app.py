import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Transactra AI",
    page_icon="💳",
    layout="wide"
)


st.markdown(
    """
    <style>

    /* Main application */

    .stApp {
        background: #071312;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background: #0F211E;
        border-right: 1px solid #1D3A35;
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 26px;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] p {
        color: #8FA7A2;
    }

    /* Main headings */

    h1 {
        font-size: 38px !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 28px !important;
    }

    h3 {
        font-size: 21px !important;
    }

    /* Metric cards */

    div[data-testid="stMetric"] {
        background: #0F211E;
        border: 1px solid #1D3A35;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.20);
    }

    div[data-testid="stMetricLabel"] {
        color: #8FA7A2;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }

    /* Buttons */

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #00D4A8;
        background: #00D4A8;
        color: #06110F;
        font-weight: 700;
        padding: 10px 18px;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #00B892;
        border-color: #00B892;
    }

    /* Info / warning / error boxes */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Dataframes */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Progress bar */

    div[data-testid="stProgressBar"] > div > div {
        background-color: #00D4A8;
    }

    /* Dividers */

    hr {
        border-color: #1D3A35;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Load Model

@st.cache_resource
def load_model():
    return joblib.load("model/fraud_detection_model.pkl")


model = load_model()

@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")


df = load_data()

@st.cache_resource
def load_evaluation():
    return joblib.load(
        "model/evaluation_results.pkl"
    )


evaluation = load_evaluation()

y_test = evaluation["y_test"]
y_prob_lr = evaluation["y_prob_lr"]
y_prob_rf = evaluation["y_prob_rf"]

lr_roc_auc = evaluation["lr_roc_auc"]
rf_roc_auc = evaluation["rf_roc_auc"]

lr_pr_auc = evaluation["lr_pr_auc"]
rf_pr_auc = evaluation["rf_pr_auc"]


# Sidebar

st.sidebar.title("💳 Transactra AI")
st.sidebar.caption("AI-Powered Fraud Detection")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Transaction Scanner",
        "Risk Analytics",
        "Batch Detection",
        "Model Performance",
        "About"
    ]
)

# Overview Page

if page == "Overview":

    st.title("💳 Transactra AI")
    st.subheader("AI-Powered Credit Card Fraud Detection")

    st.write(
        "Detect suspicious transactions using machine learning "
        "and analyze transaction risk in real time."
    )

    st.divider()

    # Dashboard metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Transactions",
        "284,807"
    )

    col2.metric(
        "Fraud Cases",
        "492"
    )

    col3.metric(
        "Fraud Rate",
        "0.17%"
    )

    col4.metric(
        "ML Model",
        "Random Forest"
    )

    st.divider()

    # Project overview

    st.subheader("How Transactra Works")

flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)

with flow_col1:
    st.markdown("### 01")
    st.markdown("**Transaction**")
    st.caption("Transaction data enters the system.")

with flow_col2:
    st.markdown("### 02")
    st.markdown("**ML Analysis**")
    st.caption("Random Forest analyzes the transaction.")

with flow_col3:
    st.markdown("### 03")
    st.markdown("**Risk Scoring**")
    st.caption("The model estimates fraud probability.")

with flow_col4:
    st.markdown("### 04")
    st.markdown("**Decision**")
    st.caption("Transactra classifies the transaction.")

    

    st.info(
        "Transactra uses machine learning to identify potentially "
        "fraudulent credit card transactions."
    )

# Transaction Scanner


if page == "Transaction Scanner":

    st.title("🔍 Transaction Scanner")

    st.write(
        "Analyze a transaction using the trained Random Forest "
        "fraud detection model."
    )

    st.divider()

    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        time_value = st.number_input(
            "Transaction Time",
            min_value=0.0,
            value=0.0
        )

        amount_value = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0
        )

    with col2:
        st.info(
            """
            **How the scanner works**

            The transaction is processed using the same
            features used during model training.

            The Random Forest model then estimates
            the probability of fraud.
            """
        )

    st.divider()

    st.subheader("Model Features")

    st.caption(
        "V1–V28 are anonymized PCA-derived features from the dataset."
    )

    feature_values = {}

    feature_cols = [f"V{i}" for i in range(1, 29)]

    columns = st.columns(4)

    for i, feature in enumerate(feature_cols):

        with columns[i % 4]:

            feature_values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=feature
            )

    st.divider()

    scan_button = st.button(
        "🚨 Scan Transaction",
        use_container_width=True
    )

    if scan_button:

        transaction = pd.DataFrame(
            [[
                time_value,
                *[
                    feature_values[f"V{i}"]
                    for i in range(1, 29)
                ],
                amount_value
            ]],
            columns=[
                "Time",
                *feature_cols,
                "Amount"
            ]
        )

        prediction = model.predict(transaction)[0]

        probability = model.predict_proba(transaction)[0][1]

        st.divider()

        st.subheader("Detection Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:

                st.error("🚨 FRAUDULENT TRANSACTION")

            else:

                st.success("✅ LEGITIMATE TRANSACTION")

        with result_col2:

            st.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

        
        # Risk Assessment

        risk_score = probability * 100

        if risk_score < 20:
            risk_level = "LOW"
            risk_message = "Transaction appears to have a low fraud risk."

        elif risk_score < 60:
            risk_level = "MEDIUM"
            risk_message = "Transaction shows some suspicious characteristics."

        else:
            risk_level = "HIGH"
            risk_message = "Transaction has a high estimated fraud risk."

        st.divider()

        st.subheader("Risk Assessment")

        risk_col1, risk_col2 = st.columns(2)

        with risk_col1:

            st.metric(
                "Risk Score",
                f"{risk_score:.1f}/100"
            )

        with risk_col2:

            if risk_level == "LOW":

                st.success(f"🟢 {risk_level} RISK")

            elif risk_level == "MEDIUM":

                st.warning(f"🟡 {risk_level} RISK")

            else:

                st.error(f"🔴 {risk_level} RISK")

        st.progress(
            min(risk_score / 100, 1.0)
        )

        st.caption(risk_message)

     
        # Risk Explanation

        st.divider()

        st.subheader("🔎 Risk Analysis")

        explanation_col1, explanation_col2 = st.columns(2)

        with explanation_col1:

            st.markdown("### Model Signal")

            if probability >= 0.60:

                st.error(
                    "The model assigns a high probability to the fraud class."
                )

            elif probability >= 0.20:

                st.warning(
                    "The model identifies a moderate level of fraud risk."
                )

            else:

                st.success(
                    "The model assigns a relatively low probability to the fraud class."
                )

        with explanation_col2:

            st.markdown("### Transaction Amount")

            if amount_value >= 1000:

                st.warning(
                    f"Transaction amount: ₹{amount_value:,.2f}"
                )

                st.caption(
                    "This is a relatively large transaction amount. "
                    "Amount alone does not determine whether a transaction is fraudulent."
                )

            else:

                st.info(
                    f"Transaction amount: ₹{amount_value:,.2f}"
                )

                st.caption(
                    "The transaction amount is included as one of the model inputs."
                )

        st.divider()

        st.caption(
            "Risk analysis is based on the trained machine learning model. "
            "It should be treated as a decision-support signal, not a definitive fraud determination."
        )

# Risk Analytics

elif page == "Risk Analytics":

    st.title("📊 Risk Analytics")

    st.write(
        "Explore transaction patterns and fraud activity "
        "across the dataset."
    )

    st.divider()


    total_transactions = len(df)
    fraud_transactions = int(df["Class"].sum())
    legitimate_transactions = total_transactions - fraud_transactions
    fraud_rate = (fraud_transactions / total_transactions) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Legitimate",
        f"{legitimate_transactions:,}"
    )

    col3.metric(
        "Fraud",
        f"{fraud_transactions:,}"
    )

    col4.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    st.divider()

    # Fraud Distribution


    st.subheader("Transaction Distribution")

    class_counts = df["Class"].value_counts()

    distribution_df = pd.DataFrame({
        "Transaction Type": ["Legitimate", "Fraud"],
        "Count": [
            class_counts.get(0, 0),
            class_counts.get(1, 0)
        ]
    })

    st.bar_chart(
        distribution_df.set_index("Transaction Type")
    )

    st.divider()

    # Transaction Amount Analysis


    st.subheader("Transaction Amount Analysis")

    amount_summary = df.groupby("Class")["Amount"].mean()

    amount_df = pd.DataFrame({
        "Transaction Type": ["Legitimate", "Fraud"],
        "Average Amount": [
            amount_summary.get(0, 0),
            amount_summary.get(1, 0)
        ]
    })

    st.bar_chart(
        amount_df.set_index("Transaction Type")
    )

    st.divider()


    st.subheader("Fraud Activity Over Time")

    time_df = df.copy()

    time_df["Time_Hours"] = time_df["Time"] / 3600

    fraud_time = (
        time_df[time_df["Class"] == 1]
        .groupby("Time_Hours")
        .size()
        .reset_index(name="Fraud Transactions")
    )

    st.line_chart(
        fraud_time.set_index("Time_Hours")
    )

# Batch Detection


elif page == "Batch Detection":

    st.title("📁 Batch Fraud Detection")

    st.write(
        "Upload a CSV file containing transaction data and "
        "scan multiple transactions at once."
    )

    st.divider()

    st.subheader("Upload Transaction File")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        batch_df = pd.read_csv(uploaded_file)

        st.write("Uploaded Data")

        st.dataframe(
            batch_df.head(10),
            use_container_width=True
        )

        required_columns = [
            "Time",
            *[f"V{i}" for i in range(1, 29)],
            "Amount"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in batch_df.columns
        ]

        if missing_columns:

            st.error(
                "The uploaded file is missing required columns."
            )

            st.write("Missing columns:")

            st.code(
                ", ".join(missing_columns)
            )

        else:

            st.success("CSV format is valid.")

            scan_batch = st.button(
                "🚨 Scan All Transactions",
                use_container_width=True
            )

            if scan_batch:

                model_input = batch_df[required_columns]

                predictions = model.predict(model_input)

                probabilities = model.predict_proba(
                    model_input
                )[:, 1]

                results_df = batch_df.copy()

                results_df["Fraud Probability"] = probabilities

                results_df["Prediction"] = predictions

                results_df["Risk Level"] = pd.cut(
                    probabilities,
                    bins=[-1, 0.20, 0.60, 1.0],
                    labels=[
                        "Low",
                        "Medium",
                        "High"
                    ]
                )

                st.divider()

                st.subheader("Batch Detection Results")

                result_col1, result_col2, result_col3 = st.columns(3)

                fraud_count = int(
                    (predictions == 1).sum()
                )

                legitimate_count = int(
                    (predictions == 0).sum()
                )

                result_col1.metric(
                    "Transactions Scanned",
                    f"{len(results_df):,}"
                )

                result_col2.metric(
                    "Fraud Detected",
                    f"{fraud_count:,}"
                )

                result_col3.metric(
                    "Legitimate",
                    f"{legitimate_count:,}"
                )

                st.dataframe(
                    results_df,
                    use_container_width=True
                )

                csv_data = results_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Detection Results",
                    data=csv_data,
                    file_name="transactra_detection_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# Model Performance


elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.write(
        "Evaluation of the machine learning models "
        "used by Transactra AI."
    )

    st.divider()

    # Model Comparison
   

    st.subheader("Model Comparison")

    comparison_path = "model/model_comparison.csv"

    if os.path.exists(comparison_path):

        comparison_df = pd.read_csv(comparison_path)

        st.dataframe(
            comparison_df.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                    "PR-AUC": "{:.4f}"
                }
            ),
            use_container_width=True
        )

    else:

        st.warning(
            "Model comparison file not found."
        )

    st.divider()



    st.subheader("ROC Curve")

    from sklearn.metrics import roc_curve

    fpr_lr, tpr_lr, _ = roc_curve(
        y_test,
        y_prob_lr
    )

    fpr_rf, tpr_rf, _ = roc_curve(
        y_test,
        y_prob_rf
    )

    fig_roc, ax_roc = plt.subplots(
        figsize=(8, 5)
    )

    ax_roc.plot(
        fpr_lr,
        tpr_lr,
        label=f"Logistic Regression (AUC = {lr_roc_auc:.3f})"
    )

    ax_roc.plot(
        fpr_rf,
        tpr_rf,
        label=f"Random Forest (AUC = {rf_roc_auc:.3f})"
    )

    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve Comparison")
    ax_roc.legend()

    st.pyplot(fig_roc)

    st.divider()

  
    # Precision-Recall Curve

    st.subheader("Precision-Recall Curve")

    from sklearn.metrics import precision_recall_curve

    precision_lr, recall_lr, _ = precision_recall_curve(
        y_test,
        y_prob_lr
    )

    precision_rf, recall_rf, _ = precision_recall_curve(
        y_test,
        y_prob_rf
    )

    fig_pr, ax_pr = plt.subplots(
        figsize=(8, 5)
    )

    ax_pr.plot(
        recall_lr,
        precision_lr,
        label=f"Logistic Regression (AP = {lr_pr_auc:.3f})"
    )

    ax_pr.plot(
        recall_rf,
        precision_rf,
        label=f"Random Forest (AP = {rf_pr_auc:.3f})"
    )

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve Comparison")
    ax_pr.legend()

    st.pyplot(fig_pr)

    st.divider()

    # Feature Importance


    st.subheader("Top Feature Importances")

    importance_path = "model/feature_importance.csv"

    if os.path.exists(importance_path):

        importance_df = pd.read_csv(
            importance_path
        )

        top_features = importance_df.head(15)

        st.bar_chart(
            top_features.set_index("Feature")[
                "Importance"
            ]
        )

        st.caption(
            "The V1–V28 variables are anonymized PCA-derived "
            "features from the original dataset."
        )

    else:

        st.warning(
            "Feature importance file not found."
        )

# About


elif page == "About":

    st.title("💳 About Transactra AI")

    st.write(
        "Transactra AI is a machine-learning powered "
        "credit card fraud detection platform designed "
        "to identify potentially suspicious transactions."
    )

    st.divider()

    st.subheader("🎯 Project Objective")

    st.write(
        "The objective of Transactra AI is to analyze "
        "transaction patterns and use machine learning "
        "to estimate the probability that a transaction "
        "belongs to the fraud class."
    )

    st.divider()

    st.subheader("🧠 Machine Learning Pipeline")

    st.markdown(
        """
        **1. Data Collection**  
        Credit card transaction dataset

        **2. Exploratory Data Analysis**  
        Transaction distributions, fraud imbalance,
        amount analysis and correlations

        **3. Preprocessing**  
        Train/test splitting and feature preparation

        **4. Model Training**  
        Logistic Regression and Random Forest

        **5. Model Evaluation**  
        Precision, Recall, F1, ROC-AUC and PR-AUC

        **6. Fraud Detection**  
        Transaction-level and batch predictions
        """
    )

    st.divider()

    st.subheader("🛠️ Technology Stack")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:

        st.markdown(
            """
            ### Python

            - Pandas
            - NumPy
            - Scikit-learn
            - Joblib
            """
        )

    with tech_col2:

        st.markdown(
            """
            ### Visualization

            - Matplotlib
            - Seaborn
            - Streamlit
            """
        )

    with tech_col3:

        st.markdown(
            """
            ### Machine Learning

            - Logistic Regression
            - Random Forest
            - Classification Metrics
            """
        )

    st.divider()

    st.subheader("⚠️ Important Note")

    st.info(
        "Transactra AI provides machine-learning based "
        "risk signals. A prediction should not be treated "
        "as definitive proof that a transaction is fraudulent."
    )