import pandas as pd

def engineer_features(df: pd.DataFrame, encoder) -> pd.DataFrame:
    """
    Perform feature engineering using a pre-fitted OneHotEncoder.
    This guarantees exact categorical column shapes matching training data, avoiding crash risks.
    """
    cat_columns = ['department', 'salary']
    num_columns = [col for col in df.columns if col not in cat_columns]
    
    # Isolate categorical and numerical data
    # Resetting index to align cleanly on concatenation
    df_cat = df[cat_columns]
    df_num = df[num_columns].reset_index(drop=True)
    
    # Transform using the loaded encoder
    # Note: The encoder must have been fit with `sparse_output=False` in training
    encoded_array = encoder.transform(df_cat)
    encoded_cols = encoder.get_feature_names_out(cat_columns)
    
    df_encoded_cat = pd.DataFrame(encoded_array, columns=encoded_cols)
    
    # Combine back with numerical features
    df_final = pd.concat([df_num, df_encoded_cat], axis=1)
    
    return df_final
