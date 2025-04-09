import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import plotly.express as px
from io import StringIO
import base64

# App configuration
st.set_page_config(
    page_title="Diabetes Anomaly Detector",
    page_icon="🪺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .reportview-container .main .block-container {padding-top: 2rem;}
    .sidebar .sidebar-content {background-color: #e8f4f8;}
    h1 {color: #2a6278;}
    h2 {color: #3a7b94;}
</style>
""", unsafe_allow_html=True)

# Session state for persistence
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

# App title and description
st.title("🪺 Diabetes Patient Anomaly Detection")
st.markdown("""
This application detects unusual patient records in diabetes datasets using machine learning algorithms.
Upload your data or use the sample dataset to identify potential anomalies.
""")

# =============================================
# Data Upload & Preview Section
# =============================================
st.header("📁 Data Upload & Preview")

upload_option = st.radio(
    "Select data source:",
    ("Upload your own CSV", "Use sample dataset")
)

if upload_option == "Upload your own CSV":
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=["csv"],
        help="Upload your diabetes dataset with similar features to the sample"
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("diabetes.csv")
    st.info("Using sample diabetes dataset. You can download it to see the expected format.")
    st.download_button(
        label="Download sample data",
        data=df.to_csv(index=False),
        file_name="diabetes_sample.csv",
        mime="text/csv"
    )

if 'df' in locals():
    with st.expander("🔍 Data Preview", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("First 5 Rows")
            st.dataframe(df.head())
        with col2:
            st.subheader("Data Summary")
            st.write(df.describe())

    with st.expander("❓ Missing Values Analysis"):
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.warning("Missing values detected in your dataset!")
            st.write(missing[missing > 0])
        else:
            st.success("No missing values detected in your dataset!")

    # =============================================
    # Anomaly Detection Configuration
    # =============================================
    st.header("⚙️ Anomaly Detection Configuration")
    with st.sidebar:
        st.header("Algorithm Settings")
        algorithm = st.radio(
            "Select detection algorithm(s):",
            ("One-Class SVM", "Local Outlier Factor (LOF)", "Both")
        )
        if algorithm in ["One-Class SVM", "Both"]:
            st.subheader("SVM Parameters")
            svm_nu = st.slider("Expected anomaly fraction (nu)", 0.01, 0.5, 0.05, 0.01)
            svm_gamma = st.selectbox("Kernel coefficient (gamma)", ["auto", "scale"] + list(np.logspace(-3, 1, 5)), index=1)
        if algorithm in ["Local Outlier Factor (LOF)", "Both"]:
            st.subheader("LOF Parameters")
            lof_neighbors = st.slider("Number of neighbors", 5, 50, 20)
            lof_contamination = st.slider("Expected contamination", 0.01, 0.5, 0.05, 0.01)

    if st.button("🚀 Run Anomaly Detection"):
        st.session_state.run_detection = True

    if st.session_state.run_detection:
        with st.spinner("Analyzing data..."):
            try:
                X = df.drop(columns=['Outcome'], errors='ignore')
            except:
                X = df.copy()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            results = pd.DataFrame(index=df.index)

            if algorithm in ["One-Class SVM", "Both"]:
                svm = OneClassSVM(kernel="rbf", gamma=svm_gamma, nu=svm_nu)
                results["SVM_Anomaly"] = svm.fit_predict(X_scaled)
                results["SVM_Anomaly"] = results["SVM_Anomaly"].map({-1: "Anomalous", 1: "Normal"})

            if algorithm in ["Local Outlier Factor (LOF)", "Both"]:
                lof = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=lof_contamination)
                results["LOF_Anomaly"] = lof.fit_predict(X_scaled)
                results["LOF_Anomaly"] = results["LOF_Anomaly"].map({-1: "Anomalous", 1: "Normal"})

            df_results = pd.concat([df, results], axis=1)

            # =============================================
            # Detection Results Display
            # =============================================
            st.header("📊 Detection Results")
            st.subheader("Detection Summary")
            col1, col2 = st.columns(2)

            if "SVM_Anomaly" in df_results.columns:
                with col1:
                    st.metric("SVM Anomalies", f"{sum(df_results['SVM_Anomaly'] == 'Anomalous')}")

            if "LOF_Anomaly" in df_results.columns:
                with col2:
                    st.metric("LOF Anomalies", f"{sum(df_results['LOF_Anomaly'] == 'Anomalous')}")

            st.subheader("📈 Visual Exploration")
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis feature", X.columns, index=0)
            with col2:
                y_axis = st.selectbox("Y-axis feature", X.columns, index=1)

            anomaly_type = st.radio(
                "Show anomalies from:",
                [a for a in ["SVM_Anomaly", "LOF_Anomaly"] if a in df_results.columns],
                horizontal=True
            )

            tab1, tab2 = st.tabs(["Interactive Plot", "Static Plot"])
            with tab1:
                fig = px.scatter(
                    df_results, x=x_axis, y=y_axis, color=anomaly_type,
                    color_discrete_map={"Anomalous": "red", "Normal": "blue"},
                    hover_data=df.columns
                )
                st.plotly_chart(fig, use_container_width=True)
            with tab2:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df_results, x=x_axis, y=y_axis, hue=anomaly_type,
                                palette={"Anomalous": "red", "Normal": "blue"})
                plt.title(f"Anomaly Detection ({anomaly_type.replace('_', ' ')})")
                st.pyplot(plt)

            # =============================================
            # Case Review System with Notes
            # =============================================
            st.header("🔍 Case Review System")
            review_tab1, review_tab2 = st.tabs(["Anomaly Browser", "Case Notes"])

            with review_tab1:
                st.subheader("Anomaly Browser")
                col1, col2 = st.columns(2)
                with col1:
                    show_anomalies = st.radio("Show cases:", ["All", "Only Anomalies", "Only Normal"], horizontal=True)
                with col2:
                    consensus_filter = st.checkbox("Only consensus anomalies", value=False)

                review_df = df_results.copy()

                # === FIXED Filtering Logic ===
                available_cols = [col for col in ["SVM_Anomaly", "LOF_Anomaly"] if col in review_df.columns]

                if show_anomalies == "Only Anomalies":
                    if consensus_filter and all(col in review_df.columns for col in ["SVM_Anomaly", "LOF_Anomaly"]):
                        review_df = review_df[(review_df["SVM_Anomaly"] == "Anomalous") & (review_df["LOF_Anomaly"] == "Anomalous")]
                    else:
                        conditions = [review_df[col] == "Anomalous" for col in available_cols]
                        review_df = review_df[np.logical_or.reduce(conditions)]
                elif show_anomalies == "Only Normal":
                    conditions = [review_df[col] == "Normal" for col in available_cols]
                    review_df = review_df[np.logical_and.reduce(conditions)]

                st.dataframe(review_df)

            with review_tab2:
                st.subheader("📒 Add Notes to Cases")
                selected_index = st.number_input("Enter record index to add notes:", min_value=0, max_value=len(df_results)-1, value=0)
                note = st.text_area("Enter your note:")
                if "case_notes" not in st.session_state:
                    st.session_state.case_notes = {}
                if st.button("🗒️ Save Note"):
                    st.session_state.case_notes[selected_index] = note
                    st.success(f"Note added for record {selected_index}!")

                if st.session_state.case_notes:
                    notes_df = pd.DataFrame([
                        {"Index": k, "Note": v} for k, v in st.session_state.case_notes.items()
                    ])
                    st.write("🗂️ Saved Notes")
                    st.dataframe(notes_df)

                    csv = notes_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="case_notes.csv">📅 Download Notes as CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
