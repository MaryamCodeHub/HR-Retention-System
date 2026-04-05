import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform preliminary data cleaning.
    """
    # Standardize column names (handled in notebook, just ensuring consistency here)
    df = df.rename(columns={
        'Work_accident': 'work_accident',
        'average_montly_hours': 'average_monthly_hours',
        'time_spend_company': 'tenure',
        'Department': 'department'
    })
    
    # Drop duplicates if any
    df = df.drop_duplicates()
    
    return df
