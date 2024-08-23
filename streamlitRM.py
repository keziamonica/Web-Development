import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from kaggle.api.kaggle_api_extended import KaggleApi

# Configuration
st.set_page_config(
    page_title='Cervical Cancer Predictor using Machine Learning Models',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Title
st.title('Cervical Cancer Predictor using Machine Learning Models')

def download_dataset():
    api = KaggleApi()
    api.set_config_value('username', st.secrets["kaggle"]["username"])
    api.set_config_value('key', st.secrets["kaggle"]["key"])
    api.authenticate()
    
    dataset = 'ajaydabas/cervical-cancer-behavior-risk-dataset'
    path = '.'

    try:
        api.dataset_download_files(dataset, path=path, unzip=True)
        st.success('Dataset downloaded successfully!')
    except Exception as e:
        st.error(f"An error occurred: {e}")

if st.sidebar.button('Get Data', type="primary"):
    download_dataset()

try:
    df = pd.read_csv("sobar-72.csv")
    st.write(df)
except FileNotFoundError:
    st.error("Dataset file not found. Please click 'Get Data' to download it.")

# Slider
st.sidebar.subheader('Input features')
behav_sexRisk = st.sidebar.slider('Behavior Sexual Risk', 1, 15, 10)
behav_eating = st.sidebar.slider('Behavior Eating', 1, 15, 10)
behav_personalHygine = st.sidebar.slider('Behavior Personal Hygiene', 1, 15, 10)
intention_aggregation = st.sidebar.slider('Intention Aggregation', 1, 15, 10)
intention_commitment = st.sidebar.slider('Intention Commitment', 1, 15, 10)
attitude_consistency = st.sidebar.slider('Attitude Consistency', 1, 15, 10)
attitude_spontaneity = st.sidebar.slider('Attitude Spontaneity', 1, 15, 10)
norm_significantPerson = st.sidebar.slider('Norm Significant Person', 1, 15, 10)
norm_fulfillment = st.sidebar.slider('Norm Fulfillment', 1, 15, 10)
perception_vulnerability = st.sidebar.slider('Perception Vulnerability', 1, 15, 10)
perception_severity = st.sidebar.slider('Perception Severity', 1, 15, 10)
motivation_strength = st.sidebar.slider('Motivation Strength', 1, 15, 10)
motivation_willingness = st.sidebar.slider('Motivation Willingness', 1, 15, 10)
socialSupport_emotionality = st.sidebar.slider('Social Support Emotionality', 1, 15, 10)
socialSupport_appreciation = st.sidebar.slider('Social Support Appreciation', 1, 15, 10)
socialSupport_instrumental = st.sidebar.slider('Social Support Instrumental', 1, 15, 10)
empowerment_knowledge = st.sidebar.slider('Empowerment Knowledge', 1, 15, 10)
empowerment_abilities = st.sidebar.slider('Empowerment Abilities', 1, 15, 10)
empowerment_desires = st.sidebar.slider('Empowerment Desires', 1, 15, 10)


# Input from slider
input_feature = pd.DataFrame({
    'behavior_sexualRisk': [behav_sexRisk],
    'behavior_eating': [behav_eating],
    'behavior_personalHygine': [behav_personalHygine],
    'intention_aggregation': [intention_aggregation],
    'intention_commitment': [intention_commitment],
    'attitude_consistency': [attitude_consistency],
    'attitude_spontaneity': [attitude_spontaneity],
    'norm_significantPerson': [norm_significantPerson],
    'norm_fulfillment': [norm_fulfillment],
    'perception_vulnerability': [perception_vulnerability],
    'perception_severity': [perception_severity],
    'motivation_strength': [motivation_strength],
    'motivation_willingness': [motivation_willingness],
    'socialSupport_emotionality': [socialSupport_emotionality],
    'socialSupport_appreciation': [socialSupport_appreciation],
    'socialSupport_instrumental': [socialSupport_instrumental],
    'empowerment_knowledge': [empowerment_knowledge],
    'empowerment_abilities': [empowerment_abilities],
    'empowerment_desires': [empowerment_desires]
})

# Display input
st.subheader('User Input from Slider')
st.write(input_feature)

# Ensure dataset is loaded before proceeding
if 'df' in locals():
    # Predictive model
    X = df.drop('ca_cervix', axis=1)
    y = df['ca_cervix']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform PCA
    pca = PCA(n_components=10)  # Choose number of components
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Handle imbalance
    oversample = RandomOverSampler(random_state=16)
    X_train_resampled, y_train_resampled = oversample.fit_resample(X_train_pca, y_train)

    # Train models
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train_resampled, y_train_resampled)

    model_lr = LogisticRegression(multi_class="auto", solver="liblinear")
    model_lr.fit(X_train_resampled, y_train_resampled)

    model_nb = GaussianNB()
    model_nb.fit(X_train_resampled, y_train_resampled)

    model_dt = DecisionTreeClassifier(random_state=42)
    model_dt.fit(X_train_resampled, y_train_resampled)

    model_svm = SVC(random_state=42)
    model_svm.fit(X_train_resampled, y_train_resampled)

    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_train_resampled, y_train_resampled)

    # Predict using input features
    input_feature_scaled = scaler.transform(input_feature)
    input_feature_pca = pca.transform(input_feature_scaled)

    y_pred_rf = model_rf.predict(input_feature_pca)
    y_pred_lr = model_lr.predict(input_feature_pca)
    y_pred_nb = model_nb.predict(input_feature_pca)
    y_pred_dt = model_dt.predict(input_feature_pca)
    y_pred_svm = model_svm.predict(input_feature_pca)
    y_pred_knn = model_knn.predict(input_feature_pca)

    # EDA and prediction results
    st.subheader('Brief EDA')
    st.write('The data is grouped by the class and the variable mean is computed for each class')
    groupby_ca_cervix_mean = df.groupby('ca_cervix').mean()
    st.write(groupby_ca_cervix_mean)
    st.line_chart(groupby_ca_cervix_mean.T)

    st.subheader('Prediction Results')

    results = pd.DataFrame({
        'Model': ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'Decision Tree', 'SVM', 'KNN'],
        'Predicted Class': [
            'Positive' if y_pred_rf[0] == 1 else 'Negative',
            'Positive' if y_pred_lr[0] == 1 else 'Negative',
            'Positive' if y_pred_nb[0] == 1 else 'Negative',
            'Positive' if y_pred_dt[0] == 1 else 'Negative',
            'Positive' if y_pred_svm[0] == 1 else 'Negative',
            'Positive' if y_pred_knn[0] == 1 else 'Negative'
        ]
    })

    st.table(results)
