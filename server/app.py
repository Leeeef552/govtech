from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load your trained ML model
model = joblib.load("model/xgb_tuned.joblib")

app = FastAPI()

# Define request schema
class PredictionRequest(BaseModel):
    month: str
    town: str
    flat_type: str
    flat_model: str
    storey_range: str
    floor_area_sqm: int
    lease_commence_date: int

# Define response schema
class PredictionResponse(BaseModel):
    predicted_price: float


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    # Convert request into dict
    input_dict = data.dict()
    # Preprocess into feature DataFrame
    X = preprocess(input_dict)
    # Predict
    prediction = model.predict(X)[0]
    return {"predicted_price": float(prediction)}


def preprocess(variables: dict):
    """
    Preprocesses input variables for model prediction.
    Parameters:
        variables : dict
            Dictionary containing:
                - month: str, transaction month in format 'YYYY-MM'
                - town: str, name of the town
                - flat_type: str, type of flat
                - storey_range: str, range of storeys
                - floor_area_sqm: float, floor area in square meters
                - flat_model: str, model of the flat
                - lease_commence_date: str or int, year when lease commenced (e.g., '1990' or 1990)
    Returns:
        dict: Encoded dictionary ready for model prediction
    """
    # Convert input dict to DataFrame
    df = pd.DataFrame([variables])

    # Convert data types
    df['lease_commence_date'] = pd.to_datetime(df['lease_commence_date'], format='%Y')
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m')

    # Extract year and month of transaction
    df['year_of_transact'] = df['month'].dt.year
    df['month_of_transact'] = df['month'].dt.month

    # Drop original 'month' column as we have extracted its components
    df = df.drop('month', axis=1)

    # Calculate years between lease commencement and sale
    df['years_between_lease_and_sale'] = (
        df['year_of_transact'] - df['lease_commence_date'].dt.year
    )

    # Calculate age of flat and remaining lease
    df['age_of_flat'] = df['years_between_lease_and_sale']
    df['remaining_lease'] = 99 - df['age_of_flat']

    df["lease_commence_date"] = df["lease_commence_date"].dt.year.astype("Int64")

    # Define categorical columns to encode
    categorical_cols = ['town', 'flat_type', 'flat_model', 'storey_range']

    # One-hot encode categorical variables
    dummies = pd.get_dummies(
        df[categorical_cols],
        columns=categorical_cols,
        prefix=categorical_cols,
        drop_first=True
    )

    # Concatenate dummy variables with main dataframe
    df = pd.concat([df, dummies], axis=1)

    # Drop original categorical columns
    df = df.drop(columns=categorical_cols)

    # Optional: Reindex to match the exact column order from training (replace with actual expected columns)
    expected_columns = ['floor_area_sqm', 'lease_commence_date', 'year_of_transact', 'month_of_transact', 'years_between_lease_and_sale', 'age_of_flat', 'remaining_lease', 'per_square_meter', 'town_bedok', 'town_bishan', 'town_bukit batok', 'town_bukit merah', 'town_bukit panjang', 'town_bukit timah', 'town_central area', 'town_choa chu kang', 'town_clementi', 'town_geylang', 'town_hougang', 'town_jurong east', 'town_jurong west', 'town_kallang/whampoa', 'town_lim chu kang', 'town_marine parade', 'town_pasir ris', 'town_punggol', 'town_queenstown', 'town_sembawang', 'town_sengkang', 'town_serangoon', 'town_tampines', 'town_toa payoh', 'town_woodlands', 'town_yishun', 'flat_type_2-room', 'flat_type_3-room', 'flat_type_4-room', 'flat_type_5-room', 'flat_type_executive', 'flat_type_multi generation', 'flat_type_multi-generation', 'flat_model_3gen', 'flat_model_adjoined flat', 'flat_model_apartment', 'flat_model_dbss', 'flat_model_improved', 'flat_model_improved-maisonette', 'flat_model_maisonette', 'flat_model_model a', 'flat_model_model a-maisonette', 'flat_model_model a2', 'flat_model_multi generation', 'flat_model_new generation', 'flat_model_premium apartment', 'flat_model_premium apartment loft', 'flat_model_premium maisonette', 'flat_model_simplified', 'flat_model_standard', 'flat_model_terrace', 'flat_model_type s1', 'flat_model_type s2', 'storey_range_01 to 05', 'storey_range_04 to 06', 'storey_range_06 to 10', 'storey_range_07 to 09', 'storey_range_10 to 12', 'storey_range_11 to 15', 'storey_range_13 to 15', 'storey_range_16 to 18', 'storey_range_16 to 20', 'storey_range_19 to 21', 'storey_range_21 to 25', 'storey_range_22 to 24', 'storey_range_25 to 27', 'storey_range_26 to 30', 'storey_range_28 to 30', 'storey_range_31 to 33', 'storey_range_31 to 35', 'storey_range_34 to 36', 'storey_range_36 to 40', 'storey_range_37 to 39', 'storey_range_40 to 42', 'storey_range_43 to 45', 'storey_range_46 to 48', 'storey_range_49 to 51']

    # Reindex using expected columns, filling missing ones with 0
    df = df.reindex(columns=expected_columns, fill_value=0)

    # Return as dictionary (for JSON-like payload)
    return df
