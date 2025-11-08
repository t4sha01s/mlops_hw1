import streamlit as st
import requests
import json
import pandas as pd

# Конфигурация
BASE_URL = "http://localhost:5000"

st.set_page_config(page_title="ML Models Dashboard", layout="wide")
st.title("ML models management dashboard")

# Боковая панель с навигацией
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Health check", "Model classes", "Train model", "Manage models", "Predict"])

# 1. Health Check
if page == "Health check":
    st.header("Service status")
    if st.button("Check health"):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                st.success("Service is indeed healthy!")
                st.json(response.json())
            else:
                st.error("Service is not responding")
        except:
            st.error("Cannot connect to service")

# 2. Model Classes
elif page == "Model classes":
    st.header("Available model classes")
    if st.button("Load available models"):
        try:
            response = requests.get(f"{BASE_URL}/model-classes")
            models = response.json()
            
            for model_name, info in models.items():
                with st.expander(f"{model_name}"):
                    st.write(f"**Description**: {info['description']}")
                    st.write(f"**Class**: {info['class_name']}")
                    st.write("**Hyperparameters**:")
                    for param in info['hyperparameters']:
                        st.write(f"  - {param}")
        except:
            st.error("Failed to load model classes")

# 3. Train Model
elif page == "Train model":
    st.header("Train new model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox("Model Type", ["random_forest", "logistic_regression"])
        
        if model_type == "random_forest":
            n_estimators = st.slider("n_estimators", 10, 100, 10)
            max_depth = st.slider("max_depth", 3, 20, 5)
            params = {"n_estimators": n_estimators, "max_depth": max_depth}
        else:
            C = st.slider("C", 0.1, 10.0, 1.0)
            max_iter = st.slider("max_iter", 100, 1000, 100)
            params = {"C": C, "max_iter": max_iter}
    
    with col2:
        st.subheader("Training Data")
        sample_data = st.text_area("X (features)", "[[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4]]")
        labels = st.text_input("y (labels)", "[0, 0, 1]")
    
    if st.button("Train model"):
        try:
            data = {
                "model_type": model_type,
                "params": params,
                "X": json.loads(sample_data),
                "y": json.loads(labels)
            }
            
            response = requests.post(f"{BASE_URL}/models/train", 
                                   headers={"Content-Type": "application/json"},
                                   data=json.dumps(data))
            
            if response.status_code == 201:
                result = response.json()
                st.success("Model trained successfully! Let`s goo")
                st.write(f"**Model ID**: {result['model_id']}")
                st.write(f"**Metrics**: {result['metrics']}")
            else:
                st.error(f"Training failed: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# 4. Manage Models
elif page == "Manage models":
    st.header("Manage trained models")
    
    # Загружаем модели
    try:
        response = requests.get(f"{BASE_URL}/models")
        models = response.json()
        
        if models:
            st.subheader(f"Found {len(models)} models")
            
            for model in models:
                with st.expander(f"{model['id']} ({model['model_type']})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Type**: {model['model_type']}")
                        st.write(f"**Created**: {model['created_at']}")
                        st.write("**Metrics**:")
                        if model['metrics']:
                            for metric, value in model['metrics'].items():
                                st.write(f"  - {metric}: {value:.3f}")
                    
                    with col2:
                        with st.form(key=f"delete_form_{model['id']}"):
                            if st.form_submit_button("Delete"):
                                try:
                                    delete_response = requests.delete(f"{BASE_URL}/models/{model['id']}")
                                    
                                    if delete_response.status_code == 204:
                                        st.success(f"Model {model['id']} deleted successfully!")
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to delete model: {delete_response.status_code}")
                                except Exception as e:
                                    st.error(f"Error deleting model: {e}")
        else:
            st.info("No models found. Train a model first! Try out our brand new training process in tab Train model")
            
    except Exception as e:
        st.error(f"Failed to load models: {e}")

# 5. Predict
elif page == "Predict":
    st.header("Make predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_id = st.text_input("Model ID", placeholder="Enter model ID")
        input_data = st.text_area("Input data", "[[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]")
    
    with col2:
        st.subheader("Sample Model IDs")
    
    if st.button("Predict"):
        if model_id and input_data:
            try:
                data = {"X": json.loads(input_data)}
                
                response = requests.post(f"{BASE_URL}/models/{model_id}/predict",
                                       headers={"Content-Type": "application/json"},
                                       data=json.dumps(data))
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction done!")
                    st.write(f"**Predictions**: {result['predictions']}")
                    
                else:
                    st.error(f"Prediction failed: {response.text}")
                    
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter Model ID and input data")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("ML Models Dashboard v1.0")
