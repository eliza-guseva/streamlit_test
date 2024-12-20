import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ml_functions import (
    read_csv,
    preprocess_data,
    split_train_test,
    get_median,
    fill_missing_values,
    feature_engineere_all,
    select_model,
    train_model,
    categorical_cols,
    numerical_cols
)
import os

st.title("Flight Satisfaction Predictor")

def train_pipeline(df):
    with st.spinner('Training model...'):
        clean_df = preprocess_data(df)
        
        # Split data
        train_x, train_y, test_x, test_y = split_train_test(clean_df)
        
        # Fill missing values
        median_dict = {col: get_median(train_x, col) for col in numerical_cols}
        pickle.dump(median_dict, open('median_dict.pkl', 'wb'))
        train_x = fill_missing_values(train_x, median_dict)
        test_x = fill_missing_values(test_x, median_dict)
        
        # Feature engineering
        train_x, train_y = feature_engineere_all(train_x, train_y, categorical_cols, is_train=True)
        test_x, test_y = feature_engineere_all(test_x, test_y, categorical_cols, is_train=False)
        
        # Model selection and training
        best_model, best_params, best_score = select_model(train_x, train_y, test_x, test_y)
        # list parameters and metrics
        st.write(f"Best model: {best_model}")
        st.write(f"Best model parameters: {best_params}")
        st.write(f"Best model accuracy: {best_score}")
        
        best_model = train_model(best_model, train_x, train_y)
        
        st.success('Model trained successfully!')
        return best_model

def predict_pipeline(df):
    with st.spinner('Making predictions...'):
        clean_df = preprocess_data(df)
        x_data = clean_df.drop(columns=['satisfaction'])
        y_true = clean_df['satisfaction']
        
        # Load artifacts and preprocess
        median_dict = pickle.load(open('median_dict.pkl', 'rb'))
        x_data = fill_missing_values(x_data, median_dict)
        x_data, y_true = feature_engineere_all(x_data, y_true, categorical_cols, is_train=False)
        
        # Load model and predict
        best_model = pickle.load(open('best_model.pkl', 'rb'))
        predictions = best_model.predict(x_data)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)
        
        return accuracy, precision, recall, f1

# Sidebar for file upload
st.sidebar.header("Upload Data")
train_file = st.sidebar.file_uploader("Upload Training Data", type=['csv'])
test_file = st.sidebar.file_uploader("Upload Test Data", type=['csv'])

# Training section
if os.path.exists('best_model.pkl'):
    st.write("A trained model already exists. Reupload training data to retrain.")
    st.write("Or upload test data to make predictions.")
else:
    # list the files in the current directory
    st.write("No trained model found. Upload training data to train a model.")
    st.write(os.listdir('.'))

if train_file is not None:
    st.header("Training")
    train_button = st.button("Train Model")
    if train_button:
        df = read_csv(train_file)
        st.write(f"Training data shape: {df.shape}")
        model = train_pipeline(df)
        st.success("Model trained successfully!")
        st.write("Model artifacts saved:")
        st.write("- best_model.pkl")
        st.write("- median_dict.pkl")
        st.write("- ohe.pkl")
        st.write("- scaler.pkl")
        st.write("- le.pkl")

# Prediction section
if test_file is not None:
    st.header("Predictions")
    if st.button("Make Predictions"):
        try:
            df = read_csv(test_file)
            st.write(f"Test data shape: {df.shape}")
            accuracy, precision, recall, f1 = predict_pipeline(df)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                st.metric("F1 Score", f"{f1:.3f}")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

st.sidebar.info("""
    1. First upload training data and click 'Train Model'
    2. Then upload test data and click 'Make Predictions'
    """) 