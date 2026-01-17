import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(
    page_title="CVD Risk Analysis",
    page_icon="🫀",
    layout="wide"
)


# --- DATA LOADING & CLEANING ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("cleaned_cardio.csv")

    # Remove outliers
    mask = (df['ap_hi'] >= 60) & (df['ap_hi'] <= 240) & \
           (df['ap_lo'] >= 40) & (df['ap_lo'] <= 160) & \
           (df['ap_hi'] > df['ap_lo'])
    df_clean = df[mask].copy()

    # Feature Engineering
    df_clean['BMI'] = df_clean['weight'] / ((df_clean['height'] / 100) ** 2)
    df_clean['pulse_pressure'] = df_clean['ap_hi'] - df_clean['ap_lo']

    return df_clean


df = load_and_clean_data()

# --- SIDEBAR NAV ---
st.sidebar.title("🫀 CVD Prediction App")
page = st.sidebar.radio(
    "Menu",
    ["🏠 Home", "📂 Dataset", "📊 Data Analysis", "🧠 Model Training", "🩺 Health Risk Test"]
)

# --- 1. HOME PAGE ---
if page == "🏠 Home":
    st.title("🫀 Cardiovascular Disease Risk Prediction")
    st.markdown("""
    ### Welcome!
    This application allows you to predict your risk of cardiovascular disease using:
    - **Random Forest Classifier**
    - **Logistic Regression**

    You can explore the dataset, perform data analysis, train models, and test your health risk.
    """)

# --- 2. DATASET PAGE ---
elif page == "📂 Dataset":
    st.title("📂 Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Records (Cleaned)", df.shape[0])
    col2.metric("Total Features", df.shape[1])

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    with st.expander("📊 Statistical Summary"):
        st.write(df.describe())

# --- 3. DATA ANALYSIS PAGE ---
elif page == "📊 Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    tab1, tab2 = st.tabs(["Distributions", "Correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Disease Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='cardio', data=df, palette='Set2', ax=ax)
            st.pyplot(fig)

        with col2:
            st.subheader("Pulse Pressure by Disease")
            fig, ax = plt.subplots()
            sns.boxplot(x='cardio', y='pulse_pressure', data=df, ax=ax)
            st.pyplot(fig)

    with tab2:
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# --- 4. MODEL TRAINING PAGE ---
elif page == "🧠 Model Training":
    st.title("🧠 Model Training")

    # Define Features and Target
    drop_cols = ['cardio', 'id'] if 'id' in df.columns else ['cardio']
    X = df.drop(drop_cols, axis=1)
    y = df['cardio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])

    if model_choice == "Random Forest":
        st.subheader("Random Forest Parameters")
        n_trees = st.slider("Number of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth", 5, 30, 10)
        min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 5, 1)

    elif model_choice == "Logistic Regression":
        st.subheader("Logistic Regression Parameters")
        solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
        max_iter = st.slider("Max Iterations", 100, 1000, 500)

    # Train button
    if st.button("🚀 Train Model"):
        if model_choice == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                class_weight='balanced'
            )
        else:
            model = LogisticRegression(
                solver=solver,
                max_iter=max_iter,
                class_weight='balanced'
            )

        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model Accuracy: {acc:.4f}")

        # Classification report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Feature importance / coefficients
        if model_choice == "Random Forest":
            st.subheader("📈 Feature Importance")
            feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
            feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', ax=ax)
            st.pyplot(fig)
        else:
            st.subheader("📊 Feature Coefficients")
            coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
            coef_df['Abs_Coeff'] = coef_df['Coefficient'].abs()
            coef_df = coef_df.sort_values(by='Abs_Coeff', ascending=False)
            st.table(coef_df[['Feature', 'Coefficient']])

        # Save model and feature names
        joblib.dump(model, "cvd_model.pkl")
        joblib.dump(X.columns.tolist(), "feature_names.pkl")
        st.info("Model saved successfully!")

# --- 5. HEALTH RISK TEST PAGE ---
elif page == "🩺 Health Risk Test":
    st.title("🩺 Cardiovascular Health Risk Test")

    # Load the trained model
    try:
        model = joblib.load("cvd_model.pkl")
        feature_names = joblib.load("feature_names.pkl")
    except:
        st.error("⚠️ Model not found! Please train the model in the 'Model Training' tab first.")
        st.stop()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (days)", value=18000)
            gender = st.selectbox("Gender", [1, 2], help="1: Women, 2: Men")
            height = st.number_input("Height (cm)", 100, 250, 170)
            weight = st.number_input("Weight (kg)", 30, 200, 70)
            ap_hi = st.number_input("Systolic Blood Pressure", 80, 240, 120)
            ap_lo = st.number_input("Diastolic Blood Pressure", 40, 160, 80)

        with col2:
            chol = st.selectbox("Cholesterol", [1, 2, 3])
            gluc = st.selectbox("Glucose", [1, 2, 3])
            smoke = st.selectbox("Smoke", [0, 1])
            alco = st.selectbox("Alcohol", [0, 1])
            active = st.selectbox("Active", [0, 1])

        submit = st.form_submit_button("🔍 Check My Risk")

    if submit:
        # Input features
        bmi = weight / ((height / 100) ** 2)
        pulse_p = ap_hi - ap_lo
        input_data = pd.DataFrame([[age, gender, height, weight, ap_hi, ap_lo,
                                    chol, gluc, smoke, alco, active, bmi, pulse_p]],
                                  columns=feature_names)

        # Prediction & probability
        prediction = model.predict(input_data)[0]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0][1]
        else:
            prob = model.decision_function(input_data)[0]
            prob = 1 / (1 + np.exp(-prob))

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ High Risk Detected! (Probability: {prob:.2%})")
            st.write("Please consult a cardiologist for professional advice.")
        else:
            st.success(f"✅ Low Risk (Probability: {prob:.2%})")
            st.write("Maintain your healthy lifestyle!")

        # Feature explanation
        st.subheader("🧩 Feature Contributions")
        if isinstance(model, RandomForestClassifier):
            importances = model.feature_importances_
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=False)
            st.table(feat_df.head(10))
        elif isinstance(model, LogisticRegression):
            coefs = model.coef_[0]
            feat_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
            feat_df['Abs_Coeff'] = feat_df['Coefficient'].abs()
            feat_df = feat_df.sort_values(by='Abs_Coeff', ascending=False)
            st.table(feat_df[['Feature', 'Coefficient']].head(10))
