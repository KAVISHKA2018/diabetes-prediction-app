import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, 
                            accuracy_score, roc_curve, auc, precision_score, 
                            recall_score, f1_score)

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
/* Title */
.sidebar-title {
    text-align: center;
    margin-top: -2rem !important;
    font-size: 30px !important;
    font-weight: 700;
}
.main-header {
    margin-top: -4rem !important;
}
hr {
    margin-top: -0.75rem !important;
}
.metric-card {
    background-color: #f0f7ff;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #2E86AB;
}
.metric-label {
    font-size: 1rem;
    color: #6c757d;
}
.sub-header {
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# Load the model and scaler
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'model.pkl' is in the root directory.")
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except FileNotFoundError:
        st.error("Scaler file not found! Please ensure 'scaler.pkl' is in the root directory.")
        return None

model = load_model()
scaler = load_scaler()

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/diabetes.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'diabetes.csv' is in the 'data/' folder.")
        return None

df = load_data()

# Stop execution if critical files are missing
if model is None or scaler is None or df is None:
    st.stop()


# Create a custom navigation in sidebar
st.sidebar.markdown("<h2 class='sidebar-title'>Diabetes Guard</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Data Exploration", "Visualizations", "Model Prediction", "Model Performance"]
)

# Add model info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Information")
st.sidebar.info("""
**Model:** Random Forest Classifier  
**Trees:** 100  
**Test Accuracy:** 78.57%  
**CV Accuracy:** 77.86%  
**Features:** 8 medical parameters
""")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.8rem;'>
    <p>Developed by MADUWANTHA J A D K</p>
    <p>ITBIN-2110-0067</p>
</div>
""", unsafe_allow_html=True)


# ***************************************************
# HOME PAGE
# ***************************************************
if page == "Home":
    st.markdown("<h1 class='main-header'>Welcome to Diabetes Guard!</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Two columns: description (left) and image (right)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        This interactive web application predicts whether a person is **diabetic** or **non-diabetic** 
        based on medical diagnostic measurements.
        
        **Key Features:**
        - Predict diabetes risk using 8 medical parameters
        - Explore the PIMA Indians Diabetes Dataset
        - Visualize data patterns and correlations
        - View model performance metrics
        - User-friendly interface with real-time predictions
        
        **Machine Learning Model:**
        - Algorithm: Random Forest Classifier
        - Number of Trees: 100
        - Dataset: PIMA Indians Diabetes Database
        - Accuracy: 78.57% on test data
        - Cross-Validation: 77.86% (5-fold CV)
        """)

    with col2:
        st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
        try:
            st.image("diabetesPrediction.jpg", use_container_width=True)
        except:
            st.info("Image not found. Add 'diabetesPrediction.jpg' to display it here.")

    # Display dataset overview with custom cards
    st.markdown("### Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{df.shape[0]}</div>
            <div class='metric-label'>Total Patients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{df.shape[1]-1}</div>
            <div class='metric-label'>Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        diabetic_count = int(df['Outcome'].sum())
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{diabetic_count}</div>
            <div class='metric-label'>Diabetic Cases</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        non_diabetic = len(df) - diabetic_count
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{non_diabetic}</div>
            <div class='metric-label'>Non-Diabetic Cases</div>
        </div>
        """, unsafe_allow_html=True)

    # About Dataset
    st.markdown("### About the Dataset")
    st.markdown("""
    The **PIMA Indians Diabetes Database** is originally from the National Institute of Diabetes 
    and Digestive and Kidney Diseases. The dataset contains diagnostic measurements for **768 female patients** 
    of Pima Indian heritage, aged **21 years or older**.
    
    **Dataset Characteristics:**
    - 768 total samples
    - 8 medical predictor variables
    - Binary classification (Diabetic vs Non-Diabetic)
    - No missing values (zeros handled during preprocessing)
    - Class distribution: 65% Non-Diabetic, 35% Diabetic
    """)
    
    # Feature Information
    st.markdown("### Features Used for Prediction")
    
    feature_info = pd.DataFrame({
        'Feature': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        'Description': [
            'Number of times pregnant',
            'Plasma glucose concentration (mg/dL)',
            'Diastolic blood pressure (mm Hg)',
            'Triceps skin fold thickness (mm)',
            '2-Hour serum insulin (mu U/ml)',
            'Body mass index (weight in kg/(height in m)²)',
            'Diabetes heredity function score',
            'Age in years'
        ],
        'Type': ['Integer', 'Integer', 'Integer', 'Integer', 
                'Integer', 'Float', 'Float', 'Integer']
    })
    
    st.dataframe(feature_info, use_container_width=True, hide_index=True)
    
    # How to Use
    st.markdown("### How to Use This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Data Exploration**
        - View dataset statistics
        - Apply interactive filters
        - Explore data distributions
        
        **2. Visualizations**
        - View correlation heatmaps
        - Analyze feature distributions
        - Identify patterns and outliers
        """)
    
    with col2:
        st.markdown("""
        **3. Model Prediction**
        - Input your medical data
        - Get real-time predictions
        - View prediction confidence
        
        **4. Model Performance**
        - Check accuracy metrics
        - View confusion matrix
        - Analyze ROC curves
        """)
    
    # Medical Disclaimer
    st.warning("""
    **Medical Disclaimer:** This application is for educational and demonstration purposes only. 
    The predictions should NOT be used as a substitute for professional medical advice, diagnosis, 
    or treatment. Always consult with a qualified healthcare provider for medical concerns.
    """)


# ***************************************************
# DATA EXPLORATION PAGE
# ***************************************************
elif page == "Data Exploration":
    st.markdown("<h1 class='main-header'>Data Exploration</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Statistical Summary", "Filter Data"])
    
    # Tab 1: Dataset Overview
    with tab1:
        st.markdown("### Dataset Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
            st.write(f"**Duplicates:** {df.duplicated().sum()}")
        
        with col2:
            st.write("**Data Types:**")
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count} columns")
        
        st.markdown("### Dataset Columns")
        columns_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.count(),
            "Unique Values": [df[col].nunique() for col in df.columns]
        })
        st.dataframe(columns_df, use_container_width=True, hide_index=True)
        
        st.markdown("### Sample Data")
        n_rows = st.slider("Number of rows to display", 5, 50, 10)
        st.dataframe(df.head(n_rows), use_container_width=True)
        
        st.markdown("### Target Distribution")
        outcome_counts = df['Outcome'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Outcome Counts:**")
            st.write(f"Non-Diabetic (0): {outcome_counts[0]} ({outcome_counts[0]/len(df)*100:.1f}%)")
            st.write(f"Diabetic (1): {outcome_counts[1]} ({outcome_counts[1]/len(df)*100:.1f}%)")

            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#2ecc71', '#e74c3c']
            ax.pie(outcome_counts, labels=['Non-Diabetic', 'Diabetic'], 
                  autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize': 15})
            ax.set_title('Class Distribution', fontsize=20)
            st.pyplot(fig)
            plt.close()
            
  
    
    # Tab 2: Statistical Summary
    with tab2:
        st.markdown("### Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)
        
        st.markdown("### Group Statistics by Outcome")
        group_stats = df.groupby('Outcome').agg(['mean', 'median', 'std']).round(2)
        st.dataframe(group_stats, use_container_width=True)
        
        st.markdown("### Correlation with Outcome")
        correlation = df.corr()['Outcome'].drop('Outcome').sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['green' if x > 0 else 'red' for x in correlation.values]
        correlation.plot(kind='barh', color=colors, ax=ax, alpha=0.7)
        ax.set_title('Feature Correlation with Diabetes Outcome', fontweight='bold')
        ax.set_xlabel('Correlation Coefficient')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # Tab 3: Filter Data
    with tab3:
        st.markdown("### Interactive Data Filtering")
        
        filtered_df = df.copy()

        # Numeric Filters
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if "Outcome" in numeric_cols:
            numeric_cols.remove("Outcome")

        st.markdown("#### Adjust Feature Ranges")
        
        for i in range(0, len(numeric_cols), 2):
            cols = st.columns([1, 1], gap="large")
            
            for j in range(2):
                if i + j < len(numeric_cols):
                    col_name = numeric_cols[i + j]
                    col_min = float(df[col_name].min())
                    col_max = float(df[col_name].max())

                    if col_name in ["BMI", "DiabetesPedigreeFunction"]:
                        step_value = 0.1
                        slider_min = round(col_min, 1)
                        slider_max = round(col_max, 1)
                        slider_default = (slider_min, slider_max)
                    else:
                        step_value = 1
                        slider_min = int(np.floor(col_min))
                        slider_max = int(np.ceil(col_max))
                        slider_default = (slider_min, slider_max)

                    selected_range = cols[j].slider(
                        f"{col_name} range",
                        min_value=slider_min,
                        max_value=slider_max,
                        value=slider_default,
                        step=step_value
                    )

                    filtered_df = filtered_df[
                        (filtered_df[col_name] >= selected_range[0]) &
                        (filtered_df[col_name] <= selected_range[1])
                    ]

        # Outcome Filter
        st.markdown("#### Filter by Outcome")
        outcome_option = st.radio(
            "Select outcome type:",
            options=["All", "Non-Diabetic (0)", "Diabetic (1)"],
            horizontal=True
        )

        if outcome_option == "Non-Diabetic (0)":
            filtered_df = filtered_df[filtered_df["Outcome"] == 0]
        elif outcome_option == "Diabetic (1)":
            filtered_df = filtered_df[filtered_df["Outcome"] == 1]

        # Display Filtered Data
        st.markdown("### Filtered Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", filtered_df.shape[0])
        col2.metric("Diabetic", int(filtered_df[filtered_df['Outcome']==1].shape[0]))
        col3.metric("Non-Diabetic", int(filtered_df[filtered_df['Outcome']==0].shape[0]))
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name='filtered_diabetes_data.csv',
            mime='text/csv',
        )


# ***************************************************
# VISUALIZATIONS PAGE
# ***************************************************
elif page == "Visualizations":
    st.markdown("<h1 class='main-header'>Data Visualizations</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.info("""
    This section provides comprehensive visualizations of the PIMA Diabetes dataset to help 
    understand data distributions, correlations, and patterns that may indicate diabetes risk factors.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Comparisons", "Advanced"])
    
    # Tab 1: Distributions
    with tab1:
        st.subheader("Feature Distributions")
        
        # Feature Distributions (Histograms)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], bins=20, kde=True, color='#3498db', ax=axes[i], edgecolor='black')
            axes[i].set_title(f'{col} Distribution', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(axis='y', alpha=0.3)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Distribution by Outcome
        st.subheader("Feature Distributions by Outcome")
        
        selected_feature = st.selectbox("Select a feature to visualize", 
                                       [col for col in df.columns if col != 'Outcome'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            df[df['Outcome']==0][selected_feature].hist(bins=20, alpha=0.6, label='Non-Diabetic', 
                                                        color='green', edgecolor='black', ax=ax)
            df[df['Outcome']==1][selected_feature].hist(bins=20, alpha=0.6, label='Diabetic', 
                                                        color='red', edgecolor='black', ax=ax)
            ax.set_xlabel(selected_feature, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{selected_feature} Distribution by Outcome', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            box_data = [df[df['Outcome']==0][selected_feature],
                       df[df['Outcome']==1][selected_feature]]
            bp = ax.boxplot(box_data, labels=['Non-Diabetic', 'Diabetic'],
                           patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title(f'{selected_feature} Box Plot', fontweight='bold')
            ax.set_ylabel(selected_feature)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    # Tab 2: Correlations
    with tab2:
        st.subheader("Correlation Analysis")
        
        # Correlation Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight='bold', pad=20)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Top Correlations
        st.subheader("Top Correlations with Outcome")
        outcome_corr = correlation_matrix['Outcome'].drop('Outcome').sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in outcome_corr.values]
        outcome_corr.plot(kind='barh', color=colors, ax=ax, alpha=0.7, edgecolor='black')
        ax.set_title('Feature Correlation with Diabetes Outcome', fontsize=14, fontweight='bold')
        ax.set_xlabel('Correlation Coefficient', fontsize=12)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Display correlation values
        st.dataframe(outcome_corr.to_frame('Correlation').style.background_gradient(cmap='RdYlGn', axis=0))
    
    # Tab 3: Comparisons
    with tab3:
        st.subheader("Feature Comparisons")
        
        # Scatter Plot
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Select X-axis feature", df.columns[:-1], index=1)
        with col2:
            y_feature = st.selectbox("Select Y-axis feature", df.columns[:-1], index=5)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_map = {0: '#2ecc71', 1: '#e74c3c'}
        for outcome in [0, 1]:
            mask = df['Outcome'] == outcome
            label = 'Non-Diabetic' if outcome == 0 else 'Diabetic'
            ax.scatter(df[mask][x_feature], df[mask][y_feature],
                      c=colors_map[outcome], label=label, alpha=0.6, 
                      edgecolors='black', s=50)
        
        ax.set_xlabel(x_feature, fontsize=12)
        ax.set_ylabel(y_feature, fontsize=12)
        ax.set_title(f'{x_feature} vs {y_feature}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Box Plots for All Features
        st.subheader("Box Plots by Outcome")
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        features = df.columns[:-1]
        colors = ['#2ecc71', '#e74c3c']
        
        for idx, feature in enumerate(features):
            box_data = [df[df['Outcome']==0][feature],
                       df[df['Outcome']==1][feature]]
            bp = axes[idx].boxplot(box_data, labels=['Non-Diabetic', 'Diabetic'],
                                  patch_artist=True, widths=0.6)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[idx].set_title(f'{feature}', fontweight='bold')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Feature Distributions by Diabetes Outcome', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Tab 4: Advanced
    with tab4:
        st.subheader("Advanced Visualizations")
        
        # Pairplot
        st.markdown("#### Pairwise Relationships")
        key_features = ['Glucose', 'BMI', 'Age', 'Insulin', 'Outcome']
        
        if all(f in df.columns for f in key_features):
            with st.spinner("Generating pairplot... This may take a moment."):
                pairplot_fig = sns.pairplot(df[key_features], hue='Outcome', 
                                           palette={0: '#2ecc71', 1: '#e74c3c'},
                                           diag_kind='kde', plot_kws={'alpha': 0.6})
                pairplot_fig.fig.suptitle("Pairwise Relationships of Key Features", 
                                         y=1.01, fontsize=14, fontweight='bold')
                st.pyplot(pairplot_fig)
                plt.close()
        
        st.markdown("---")


# ***************************************************
# MODEL PREDICTION PAGE
# ***************************************************
elif page == "Model Prediction":
    st.markdown("<h1 class='main-header'>Diabetes Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.info("""
    Enter your medical data below to predict diabetes risk. The model uses a trained 
    Support Vector Machine (SVM) classifier to provide predictions based on 8 key health metrics.
    """)

    # Input Form
    st.markdown("### Enter Your Medical Data")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", 
                                         min_value=0, max_value=20, value=1, 
                                         help="Number of times pregnant")
            glucose = st.number_input("Glucose Level (mg/dL)", 
                                     min_value=0, max_value=200, value=120,
                                     help="Plasma glucose concentration")
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", 
                                            min_value=0, max_value=150, value=70,
                                            help="Diastolic blood pressure")
            skin_thickness = st.number_input("Skin Thickness (mm)", 
                                            min_value=0, max_value=100, value=20,
                                            help="Triceps skin fold thickness")
        
        with col2:
            insulin = st.number_input("Insulin Level (mu U/ml)", 
                                     min_value=0, max_value=900, value=79,
                                     help="2-Hour serum insulin")
            bmi = st.number_input("BMI", 
                                 min_value=0.0, max_value=70.0, value=25.0, step=0.1,
                                 help="Body Mass Index")
            dpf = st.number_input("Diabetes Pedigree Function", 
                                 min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                                 help="Diabetes heredity function")
            age = st.number_input("Age (years)",
                                 min_value=10, max_value=100, value=33,
                                 help="Age in years")
        
        submitted = st.form_submit_button("Get Prediction", use_container_width=True)
    
    if submitted:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })

        st.markdown("### Input Summary")
        st.dataframe(input_data, use_container_width=True)

        # Standardize input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get confidence score
        confidence = None
        if hasattr(model, 'decision_function'):
            decision_score = model.decision_function(input_scaled)[0]
            confidence = abs(decision_score)
        elif hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
            confidence = max(proba)

        # Display result    
        st.markdown("---")
        st.subheader("Prediction Result")

        # Determine risk level based on prediction and confidence
        if confidence is None:
            confidence = 0.5  # fallback if not available

        if prediction == 0:
            risk_level = confidence * 0.4  # lower for non-diabetic
        else:
            risk_level = confidence * 0.9 + 0.1  # higher for diabetic

        # Define progress bar color and label
        if risk_level < 0.5:
            color = "#2ecc71"  # green
            risk_text = "Low Risk"
        else:
            color = "#e74c3c"  # red
            risk_text = "High Risk"

        # Show prediction result with custom background colors
        if prediction == 0:
            st.markdown("""
            <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; 
                        border-left: 5px solid #28a745; margin: 10px 0;'>
                <h2 style='color: #155724; margin: 0;'> NON-DIABETIC</h2>
                <p style='color: #155724; margin: 10px 0 0 0; font-size: 16px;'>
                    Lower diabetes risk detected based on the provided measurements.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px; 
                        border-left: 5px solid #dc3545; margin: 10px 0;'>
                <h2 style='color: #721c24; margin: 0;'> DIABETIC</h2>
                <p style='color: #721c24; margin: 10px 0 0 0; font-size: 16px;'>
                    Higher diabetes risk detected. Please consult a healthcare provider.
                </p>
            </div>
            """, unsafe_allow_html=True)
            

        # Display progress bar as a styled HTML component
        st.markdown(f"""
        <div style='margin-top: 1.2rem;'>
            <div style='font-weight:600;'>Diabetes Risk Level: {risk_text}</div>
            <div style='height: 25px; width: 100%; background-color: #e0e0e0; border-radius: 10px;'>
                <div style='height: 25px; width: {risk_level*100:.1f}%; background-color: {color};
                            border-radius: 10px; text-align: center; color: white; font-weight: bold;'>
                    {risk_level*100:.0f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence metric (optional)
        st.metric("Confidence", f"{confidence:.2%}")

        st.info("""
        **Disclaimer:** This prediction is based on a machine learning model trained on the 
        PIMA Indians Diabetes Dataset. It should not replace professional medical diagnosis. 
        Always consult with a healthcare provider for accurate medical assessment.
        """)


# ***************************************************
# MODEL PERFORMANCE PAGE
# ***************************************************
elif page == "Model Performance":
    st.markdown("<h1 class='main-header'>Model Performance</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Prepare data
    X = df.drop(columns='Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_train_pred = model.predict(X_train_scaled)

    # Calculate metrics
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Metrics", "Confusion Matrix", "ROC Curve"])
    
    # Tab 1: Overview
    with tab1:
        st.markdown("### Model Details")
        st.info("""
        **Model Used:** Random Forest Classifier  
        **Number of Trees:** 100  
        **Feature Scaler:** StandardScaler  
        **Dataset:** PIMA Indians Diabetes Dataset  
        **Training/Test Split:** 80-20 (stratified)
        """)

        st.markdown("### Performance Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Train Accuracy", f"{train_acc:.4f}")
        col2.metric("Test Accuracy", f"{test_acc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")
        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1-Score", f"{f1:.4f}")
    
    # Tab 2: Detailed Metrics
    with tab2:
        st.markdown("### Classification Report")
        report = classification_report(y_test, y_pred, 
                                       target_names=['Non-Diabetic', 'Diabetic'],
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)
        
        st.markdown("### Metric Explanations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Precision:** Of the predicted positive cases, how many were actually positive?
            - High precision = Few false positives
            - Formula: TP / (TP + FP)
            
            **Recall:** Of the actual positive cases, how many did the model correctly identify?
            - High recall = Few false negatives
            - Formula: TP / (TP + FN)
            """)
        
        with col2:
            st.markdown("""
            **F1-Score:** Harmonic mean of precision and recall
            - Balances precision and recall
            - Formula: 2 * (Precision * Recall) / (Precision + Recall)
            
            **Support:** Number of samples in each class
            """)
    
    # Tab 3: Confusion Matrix
    with tab3:
        st.markdown("### Confusion Matrix - Test Set")
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, 
                                              cmap='Blues', ax=ax, values_format='d')
        ax.set_title("Confusion Matrix", fontweight='bold', fontsize=14)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("### Confusion Matrix Interpretation")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **True Positives (TP):** Correctly predicted diabetic cases  
            **True Negatives (TN):** Correctly predicted non-diabetic cases
            """)
        
        with col2:
            st.markdown("""
            **False Positives (FP):** Non-diabetic patients predicted as diabetic  
            **False Negatives (FN):** Diabetic patients predicted as non-diabetic
            """)
    
    # Tab 4: ROC Curve
    with tab4:
        st.markdown("### ROC Curve Analysis")
        
        if hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_test_scaled)
        elif hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_scores = None

        if y_scores is not None:
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2.5, 
                   label=f'ROC Curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate', fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontweight='bold')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=14)
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            st.markdown("### ROC Curve Interpretation")
            st.markdown(f"""
            **AUC Score:** {roc_auc:.4f}
            
            - **AUC = 1.0:** Perfect classifier
            - **AUC = 0.5:** Random classifier (no discrimination)
            - **AUC = 0.0:** Worst possible classifier
            
            Your model's AUC of **{roc_auc:.4f}** indicates {'excellent' if roc_auc > 0.9 else 'good' if roc_auc > 0.8 else 'acceptable'} discriminative ability.
            """)

    # Model Insights
    st.markdown("---")
    st.markdown("### Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Strengths:**
        - Strong overall accuracy on test data
        - Balanced precision and recall
        - Good generalization (minimal overfitting)
        - Fast inference time suitable for real-time predictions
        """)
    
    with col2:
        st.markdown("""
        **Recommendations:**
        - Use model for initial screening purposes
        - Combine with clinical judgment for final diagnosis
        - Monitor for data drift in new datasets
        - Consider ensemble methods for further improvement
        """)

    st.divider()
    st.caption("Diabetes Prediction Model | ML Deployment Project | Developed by MADUWANTHA J A D K | ITBIN-2110-0067")