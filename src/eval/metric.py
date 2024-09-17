"""
TODO: Enter any documentation that only people updating the metric should read here.

All columns of the solution and submission dataframes are passed to your metric, except for the Usage column.

Your metric must satisfy the following constraints:
- You must have a function named score. Kaggle's evaluation system will call that function.
- You can add your own arguments to score, but you cannot change the first three (solution, submission, and row_id_column_name).
- All arguments for score must have type annotations.
- score must return a single, finite, non-null float.
"""

import pandas as pd
import pandas.api.types
from geopy.distance import geodesic
import numpy as np

class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Calculates the weighted distance between the actual and predicted latitude/longitude points.
    
    The function computes the geodesic distance between the actual coordinates (`latitude`, `longitude`) 
    and the predicted coordinates (`latitude_predicted`, `longitude_predicted`) in meters. 
    This distance is then scaled by the provided `scaling_factor` to obtain the weighted distance.
    '''

    # TODO: You likely want to delete the row ID column, which Kaggle's system uses to align
    # the solution and submission before passing these dataframes to score().
    #del solution[row_id_column_name]
    #del submission[row_id_column_name]

    # TODO: adapt or remove this check depending on what data types make sense for your metric
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
                    
    # TODO: add additional checks appropriate for your metric. Common things to check include:
    # Non-negative inputs, the wrong number of columns in the submission, values that should be restricted to a range, etc.
    # The more errors you tag as participant visible, the easier it will be for participants to debug their submissions.

    for col in ["ID", "longitude_predicted", "latitude_predicted"]: 
        if col not in submission.columns: 
            raise ParticipantVisibleError(f'Submission is missing column {col}')

    # TODO: calculate your metric here. For the template, we'll just calculate a simple mean absolute error over all non-id columns.
    # By default, the metric doesn't have information about what columns to expect. You probably want to add the names of other
    # columns as arguments to score, like row_id_column_name, if you won't be processing all columns the same way.
    
    # Merging the predictions to the ground truth

    try:
        solution_submission = solution.merge(submission[[row_id_column_name, 'longitude_predicted', 'latitude_predicted']], on=row_id_column_name, how='left')
        solution_submission['weighted_distance'] = solution_submission.apply(calculate_distance, axis=1)
    
        weighted_distance = solution_submission['weighted_distance'].mean() / 1000.0

        return weighted_distance
    
    except:
        raise ParticipantVisibleError(f'Evaluation metric raised an unexpected error')
    

def calculate_distance(row):
    """Calculates the weighted distance between the actual and predicted lat/long points."""
    if pd.isna(row['latitude']) or pd.isna(row['latitude_predicted']):
        return np.nan
    # Calculate the geodesic distance in meters
    distance = geodesic((row['latitude'], row['longitude']), 
                        (row['latitude_predicted'], row['longitude_predicted'])).meters
    # Weight the distance by the scaling factor
    weighted_distance = distance * row['scaling_factor']
    return weighted_distance
