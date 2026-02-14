import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

# PAGE SETTINGS 
st.set_page_config(
    page_title="Student Depression Prediction System",
    layout="wide"
)

# TITLE 
st.title("Student Depression Prediction System")
st.write("Upload dataset, select model and view results.")
st.divider()

# FILE UPLOAD 
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file).dropna()

    # PREPROCESSING 
    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop("Depression", axis=1)
    y = df["Depression"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # MODEL SELECTION 
    st.subheader("Select Model")

    with st.form("model_form"):
        model_name = st.selectbox(
            "Choose ML Model",
            ["Logistic Regression", "Decision Tree", "KNN",
             "Naive Bayes", "Random Forest", "XGBoost"]
        )
        submitted = st.form_submit_button("Train Model")

    if submitted:

        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
            filename = "Logistic_Regression.pkl"
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()
            filename = "Decision_Tree.pkl"
        elif model_name == "KNN":
            model = KNeighborsClassifier()
            filename = "KNN.pkl"
        elif model_name == "Naive Bayes":
            model = GaussianNB()
            filename = "Naive_Bayes.pkl"
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
            filename = "Random_Forest.pkl"
        else:
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            filename = "XGBoost.pkl"

        # TRAIN MODEL 
        model.fit(X_train, y_train)

        os.makedirs("model", exist_ok=True)
        joblib.dump(model, os.path.join("model", filename))

        preprocessor = {
            "scaler": scaler,
            "label_encoders": label_encoders,
            "feature_columns": X.columns.tolist()
        }
        joblib.dump(preprocessor, os.path.join("model", "preprocessor.pkl"))

        st.success(f"{model_name} Trained Successfully")
        st.write(f"Saved model at: model/{filename}")

        # PREDICTIONS 
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # METRICS 
        st.subheader("Evaluation Metrics")

        metrics_df = pd.DataFrame({
            "ML Model Name": [model_name],
            "Accuracy": [round(accuracy_score(y_test, y_pred), 3)],
            "AUC": [round(roc_auc_score(y_test, y_prob), 3)],
            "Precision": [round(precision_score(y_test, y_pred), 3)],
            "Recall": [round(recall_score(y_test, y_pred), 3)],
            "F1": [round(f1_score(y_test, y_pred), 3)],
            "MCC": [round(matthews_corrcoef(y_test, y_pred), 3)]
        })

        st.dataframe(metrics_df, use_container_width=True)
        st.divider()

        # CLASSIFICATION REPORT 
        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            target_names=["Not Depressed", "Depressed"],
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        st.divider()

        # CONFUSION MATRIX 
        st.subheader(f"Confusion Matrix – {model_name}")

        cm_df = pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            index=["Not Depressed", "Depressed"],
            columns=["Not Depressed", "Depressed"]
        )

        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Blues"
            )

            fig_cm.update_layout(
                width=600,
                height=600,
                font=dict(size=24)
            )

            fig_cm.update_traces(hoverinfo="skip", hovertemplate=None)

            st.plotly_chart(
                fig_cm,
                use_container_width=False,
                config={"displayModeBar": False}
            )

        st.divider()

        # PROBABILITY DISTRIBUTION 
        st.subheader(f"Prediction Probability Distribution – {model_name}")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=y_prob,
            nbinsx=20,
            marker_color="skyblue",
            hovertemplate="Predicted Probability: %{x:.3f}<br>Number of Students: %{y}<extra></extra>"
        ))

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Predicted Probability",
            yaxis_title="Number of Students",
            width=900,
            height=450,
            bargap=0.15
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

else:
    st.info("Please upload a CSV file to begin.")
