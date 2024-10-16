import pandas as pd

# TODO: 

# make coeff for evaluation
def prepare_validation_set(df):
    # Ensure the dataframe is sorted by vessel and date
    df = df.sort_values(by=['vessel_id', 'date'])
    
    # Group by vessel_id and date, then take the last 5 days for each vessel
    last_5_days = df.groupby('vessel_id')['date'].transform(lambda x: x.nlargest(5))
    validation_set = df[df['date'].isin(last_5_days)].copy()
    
    # Define the coefficients
    coefficients = [0.3, 0.25, 0.2, 0.15, 0.1]
    
    # Add the coefficients to the validation set
    validation_set['coefficient'] = validation_set.groupby('vessel_id').cumcount(ascending=False).map(lambda x: coefficients[x])
    
    return validation_set