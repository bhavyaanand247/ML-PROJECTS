import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# LOAD MODEL


@st.cache_resource
def load_model():
    return joblib.load(
        "model/intrusion_detection_model.pkl"
    )


model = load_model()

@st.cache_resource
def load_evaluation():
    return joblib.load(
        "model/evaluation_results.pkl"
    )


evaluation = load_evaluation()

y_test = evaluation["y_test"]
y_pred_lr = evaluation["y_pred_lr"]
y_prob_lr = evaluation["y_prob_lr"]
y_pred_rf = evaluation["y_pred_rf"]
y_prob_rf = evaluation["y_prob_rf"]



@st.cache_data
def load_data():
    columns = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "label",
        "difficulty"
    ]

    data = pd.read_csv(
        "data/KDDTrain+.csv",
        header=None,
        names=columns
    )

    data["attack"] = data["label"].apply(
        lambda x: 0 if x == "normal" else 1
    )

    return data


df = load_data()

X = df.drop(
    columns=["label", "attack", "difficulty"]
)


st.markdown(
    """
    <style>

    .main {
        background-color: #07111F;
    }

    .block-container {
        padding-top: 2rem;
    }

    .metric-card {
        background: #0D1B2A;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #17324D;
    }

    .hero {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0D1B2A,
            #07111F
        );
        border: 1px solid #17324D;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        color: #00E5FF;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #9FB3C8;
        font-size: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# SIDEBAR


st.sidebar.markdown(
    "# 🛡️ NetGuard AI"
)

st.sidebar.caption(
    "Network Intrusion Detection System"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Intrusion Detection",
        "Traffic Analytics",
        "Batch Detection",
        "Model Performance",
        "About"
    ]
)

# OVERVIEW


if page == "Overview":

    st.markdown(
        """
        <div class="hero">

        <div class="hero-title">
        🛡️ NetGuard AI
        </div>

        <div class="hero-subtitle">
        Machine Learning powered Network Intrusion Detection System
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    total_records = len(df)
    attack_records = int(df["attack"].sum())
    normal_records = total_records - attack_records
    attack_rate = attack_records / total_records * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Network Records",
            f"{total_records:,}"
        )

    with col2:
        st.metric(
            "Normal Traffic",
            f"{normal_records:,}"
        )

    with col3:
        st.metric(
            "Detected Attacks",
            f"{attack_records:,}"
        )

    with col4:
        st.metric(
            "Attack Rate",
            f"{attack_rate:.1f}%"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        attack_data = (
            df["attack"]
            .value_counts()
            .rename(index={
                0: "Normal",
                1: "Attack"
            })
            .reset_index()
        )

        attack_data.columns = [
            "Traffic Type",
            "Count"
        ]

        fig = px.pie(
            attack_data,
            names="Traffic Type",
            values="Count",
            title="Normal vs Attack Traffic",
            hole=0.45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        top_attacks = (
            df["label"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        top_attacks.columns = [
            "Attack Type",
            "Count"
        ]

        fig = px.bar(
            top_attacks,
            x="Count",
            y="Attack Type",
            orientation="h",
            title="Top Network Traffic Types"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# INTRUSION DETECTION


elif page == "Intrusion Detection":

    st.title("🔍 Intrusion Detection")

    st.write(
        "Analyze a network connection using the trained "
        "machine learning model."
    )

    st.markdown("---")

    st.subheader("🌐 Network Connection Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        protocol_type = st.selectbox(
            "Protocol",
            ["tcp", "udp", "icmp"]
        )

        service = st.selectbox(
            "Service",
            sorted(df["service"].unique())
        )

    with col2:

        flag = st.selectbox(
            "Connection Flag",
            sorted(df["flag"].unique())
        )

        duration = st.number_input(
            "Connection Duration",
            min_value=0,
            value=0
        )

    with col3:

        src_bytes = st.number_input(
            "Source Bytes",
            min_value=0,
            value=100
        )

        dst_bytes = st.number_input(
            "Destination Bytes",
            min_value=0,
            value=100
        )

    st.markdown("---")

    st.subheader("⚙️ Detection")

    if st.button(
        "🛡️ Analyze Connection",
        use_container_width=True
    ):

        # Using median values from the training dataset
        # for the remaining numerical features.

        input_data = {}

        for column in X.columns:

            if column in [
                "protocol_type",
                "service",
                "flag"
            ]:

                continue

            input_data[column] = df[column].median()

        input_data["duration"] = duration
        input_data["src_bytes"] = src_bytes
        input_data["dst_bytes"] = dst_bytes

        input_data["protocol_type"] = protocol_type
        input_data["service"] = service
        input_data["flag"] = flag

        input_df = pd.DataFrame(
            [input_data]
        )

        prediction = model.predict(
            input_df
        )[0]

        probability = model.predict_proba(
            input_df
        )[0][1]

        st.markdown("---")

        if prediction == 1:

            st.error(
                "🚨 ATTACK DETECTED"
            )

            st.metric(
                "Attack Probability",
                f"{probability * 100:.2f}%"
            )

            st.warning(
                "The model classified this network "
                "connection as potentially malicious."
            )

        else:

            st.success(
                "🟢 NORMAL TRAFFIC"
            )

            st.metric(
                "Attack Probability",
                f"{probability * 100:.2f}%"
            )

            st.info(
                "The model classified this network "
                "connection as normal."
            )


# TRAFFIC ANALYTICS


elif page == "Traffic Analytics":

    st.title("📊 Traffic Analytics")

    service_counts = (
        df["service"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    service_counts.columns = [
        "Service",
        "Count"
    ]

    fig = px.bar(
        service_counts,
        x="Count",
        y="Service",
        orientation="h",
        title="Most Common Network Services"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    protocol_counts = (
        df["protocol_type"]
        .value_counts()
        .reset_index()
    )

    protocol_counts.columns = [
        "Protocol",
        "Count"
    ]

    fig = px.pie(
        protocol_counts,
        names="Protocol",
        values="Count",
        title="Protocol Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# BATCH DETECTION


elif page == "Batch Detection":

    st.title("📁 Batch Detection")

    st.write(
        "Upload a CSV file containing network traffic "
        "and classify multiple connections at once."
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_df = pd.read_csv(uploaded_file)

        st.subheader("📋 Uploaded Data")

        st.write(
            f"Rows: {len(uploaded_df):,} | "
            f"Columns: {len(uploaded_df.columns)}"
        )

        st.dataframe(
            uploaded_df.head(10),
            use_container_width=True
        )

        
        required_features = [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate"
        ]

        missing_features = [
            feature
            for feature in required_features
            if feature not in uploaded_df.columns
        ]

        if missing_features:

            st.error(
                "❌ The uploaded CSV is missing required features."
            )

            st.write("Missing features:")

            st.code(
                ", ".join(missing_features)
            )

        else:

            prediction_data = uploaded_df[
                required_features
            ].copy()

            if st.button(
                "🛡️ Analyze All Connections",
                use_container_width=True
            ):

                predictions = model.predict(
                    prediction_data
                )

                probabilities = model.predict_proba(
                    prediction_data
                )[:, 1]

                results = uploaded_df.copy()

                results["Prediction"] = np.where(
                    predictions == 1,
                    "Attack",
                    "Normal"
                )

                results["Attack Probability"] = (
                    probabilities * 100
                ).round(2)

                total_connections = len(results)

                attack_count = int(
                    (predictions == 1).sum()
                )

                normal_count = int(
                    (predictions == 0).sum()
                )

                attack_percentage = (
                    attack_count / total_connections * 100
                )

                st.markdown("---")

                st.subheader(
                    "🚨 Detection Results"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Connections",
                        f"{total_connections:,}"
                    )

                with col2:
                    st.metric(
                        "Normal",
                        f"{normal_count:,}"
                    )

                with col3:
                    st.metric(
                        "Attacks",
                        f"{attack_count:,}"
                    )

                with col4:
                    st.metric(
                        "Attack Rate",
                        f"{attack_percentage:.1f}%"
                    )

                st.markdown("---")

                st.subheader(
                    "📊 Classification Results"
                )

                result_counts = (
                    results["Prediction"]
                    .value_counts()
                    .reset_index()
                )

                result_counts.columns = [
                    "Prediction",
                    "Count"
                ]

                fig = px.pie(
                    result_counts,
                    names="Prediction",
                    values="Count",
                    title="Normal vs Attack"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.subheader(
                    "🔍 Connection-Level Results"
                )

                st.dataframe(
                    results,
                    use_container_width=True
                )

                csv_data = results.to_csv(
                    index=False
                )

                st.download_button(
                    label="⬇️ Download Detection Results",
                    data=csv_data,
                    file_name="netguard_detection_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# MODEL PERFORMANCE


elif page == "Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Evaluation of the machine learning models "
        "used for network intrusion detection."
    )

    st.markdown("---")

    
    # MODEL METRICS

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score
    )

    # Logistic Regression metrics
    lr_accuracy = accuracy_score(
        y_test,
        y_pred_lr
    )

    lr_precision = precision_score(
        y_test,
        y_pred_lr
    )

    lr_recall = recall_score(
        y_test,
        y_pred_lr
    )

    lr_f1 = f1_score(
        y_test,
        y_pred_lr
    )

    lr_auc = roc_auc_score(
        y_test,
        y_prob_lr
    )

    # Random Forest metrics
    rf_accuracy = accuracy_score(
        y_test,
        y_pred_rf
    )

    rf_precision = precision_score(
        y_test,
        y_pred_rf
    )

    rf_recall = recall_score(
        y_test,
        y_pred_rf
    )

    rf_f1 = f1_score(
        y_test,
        y_pred_rf
    )

    rf_auc = roc_auc_score(
        y_test,
        y_prob_rf
    )


    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],
        "Accuracy": [
            lr_accuracy,
            rf_accuracy
        ],
        "Precision": [
            lr_precision,
            rf_precision
        ],
        "Recall": [
            lr_recall,
            rf_recall
        ],
        "F1 Score": [
            lr_f1,
            rf_f1
        ],
        "ROC-AUC": [
            lr_auc,
            rf_auc
        ]
    })

    st.subheader("📊 Model Comparison")

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        }),
        use_container_width=True
    )


    st.subheader("🌲 Random Forest Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{rf_accuracy:.3f}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{rf_precision:.3f}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{rf_recall:.3f}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{rf_f1:.3f}"
        )

    with col5:
        st.metric(
            "ROC-AUC",
            f"{rf_auc:.3f}"
        )


    st.markdown("---")

    st.subheader("📈 ROC Curve")

    from sklearn.metrics import roc_curve

    fpr_lr, tpr_lr, _ = roc_curve(
        y_test,
        y_prob_lr
    )

    fpr_rf, tpr_rf, _ = roc_curve(
        y_test,
        y_prob_rf
    )

    roc_df = pd.DataFrame({
        "False Positive Rate": fpr_lr,
        "Logistic Regression": tpr_lr
    })

    fig = px.line(
        roc_df,
        x="False Positive Rate",
        y="Logistic Regression",
        title="ROC Curve"
    )

    fig.add_scatter(
        x=fpr_rf,
        y=tpr_rf,
        mode="lines",
        name="Random Forest"
    )

    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(dash="dash")
    )

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

 

    st.subheader(
        "🔲 Random Forest Confusion Matrix"
    )

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(
        y_test,
        y_pred_rf
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Attack"],
        columns=["Predicted Normal", "Predicted Attack"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        title="Random Forest Confusion Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # FEATURE IMPORTANCE

    st.subheader(
        "🎯 Top Features"
    )

    feature_names = (
        model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        model
        .named_steps["classifier"]
        .feature_importances_
    )

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feature_importance = (
        feature_importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(15)
        .sort_values("Importance")
    )

    fig = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Features Used by Random Forest"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif page == "About":

    st.title("🛡️ About NetGuard AI")

    st.markdown(
        """
        ## Network Intrusion Detection System

        **NetGuard AI** is a machine learning based system
        designed to classify network connections as either
        normal traffic or potential attacks.

        ### 🔬 Machine Learning

        The project compares:

        - Logistic Regression
        - Random Forest

        The models are evaluated using:

        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC-AUC
        - Confusion Matrix

        ### 🧠 Dataset

        The project uses the **NSL-KDD dataset**, containing
        network traffic records with multiple attack types.

        For the main classification task, the original labels
        are converted into:

        **0 → Normal Traffic**

        **1 → Attack Traffic**

        ### ⚙️ Technology Stack

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Joblib
        - Streamlit
        - Plotly

        ### 🚨 Important Note

        NetGuard AI is an educational and portfolio project.
        Its predictions should not be treated as a replacement
        for a production-grade cybersecurity monitoring system.
        """
    )

    st.markdown("---")

    st.caption(
        "NetGuard AI • Machine Learning Network Intrusion Detection"
    )