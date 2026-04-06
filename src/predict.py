import pickle
import pandas as pd
import shap
import os

from src.data_preprocessing import clean_data
from src.feature_engineering import engineer_features

def load_model(model_path: str, encoder_path: str, features_path: str):
    """
    Loads the trained model, SHAP explainer, fitted OneHotEncoder, and feature columns list.
    """
    if not os.path.exists(model_path) or not os.path.exists(encoder_path) or not os.path.exists(features_path):
        return None, None, None, None
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
        
    with open(features_path, 'rb') as f:
        feature_cols = pickle.load(f)
        
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = None
        
    return model, explainer, encoder, feature_cols

def predict_employee_attrition(model, explainer, encoder, feature_cols, employee_data: dict) -> dict:
    """
    Runs the pipeline: clean -> encode -> strict reindex -> predict -> explain.
    """
    df = pd.DataFrame([employee_data])
    
    # Preprocess & Engineer (Uses trained encoder here)
    df = clean_data(df)
    df_features = engineer_features(df, encoder)
    
    # CRITICAL: Enforce exact feature column order matching the trained model 
    df_features = df_features.reindex(columns=feature_cols, fill_value=0)
    
    # Predict Probability
    try:
        prediction_prob = model.predict_proba(df_features)[0][1]
    except AttributeError:
        prediction_prob = float(model.predict(df_features)[0])
        
    # Generate Explainability (SHAP)
    top_factors = []
    if explainer is not None:
        try:
            import numpy as np
            shap_values = explainer.shap_values(df_features)
            
            if isinstance(shap_values, list):
                vals = shap_values[1][0]
            else:
                sv = np.array(shap_values)
                if len(sv.shape) == 3:
                    vals = sv[0, :, 1]  # shape (samples, features, classes)
                elif len(sv.shape) == 2:
                    vals = sv[0]
                else:
                    vals = sv.flatten()
                    
            vals = [float(v) for v in vals] # Ensure 1D python floats
            
            feature_names = df_features.columns.tolist()
            
            impacts = [{"feature": f, "impact": abs(v), "direction": "Positive" if v > 0 else "Negative"} 
                       for f, v in zip(feature_names, vals)]
            
            impacts.sort(key=lambda x: x['impact'], reverse=True)
            top_factors = impacts[:3] 
        except Exception as e:
             # Output exact error for debugging
            top_factors = [{"error": f"SHAP Error: {str(e)}"}]

    return {
        "attrition_probability": round(prediction_prob, 4),
        "risk_level": "High" if prediction_prob > 0.5 else "Low",
        "top_driving_factors": top_factors
    }
