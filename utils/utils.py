import pandas as pd

## preprocess function to prepare payload for prediction
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

## restriction to arguements for prediction by llm
def get_valid_values():
    return {
        "towns": [
            "woodlands", "jurong west", "tampines", "yishun", "bedok",
            "sengkang", "hougang", "ang mo kio", "bukit batok", "bukit merah",
            "choa chu kang", "pasir ris", "bukit panjang", "toa payoh",
            "kallang/whampoa", "geylang", "queenstown", "punggol", "clementi",
            "jurong east", "sembawang", "serangoon", "bishan", "marine parade",
            "central area", "bukit timah"
        ],
        "flat_types": [
            "4-room", "3-room", "5-room", "executive",
            "2-room", "1-room", "multi-generation"
        ],
        "storey_ranges": [
            "04 to 06", "07 to 09", "01 to 03", "10 to 12", "13 to 15",
            "01 to 05", "06 to 10", "16 to 18", "11 to 15", "19 to 21",
            "22 to 24", "16 to 20", "25 to 27", "28 to 30", "21 to 25",
            "26 to 30", "34 to 36", "37 to 39", "31 to 33", "40 to 42",
            "36 to 40", "31 to 35"
        ],
        "flat_models": [
            "model a", "improved", "new generation", "premium apartment",
            "simplified", "apartment", "maisonette", "standard", "dbss",
            "model a2", "model a-maisonette", "adjoined flat", "type s1",
            "2-room", "type s2", "premium apartment loft", "terrace",
            "multi generation", "3gen", "improved-maisonette", "premium maisonette"
        ],
        "min_area": 31,
        "max_area": 266
    }


def get_defaults():
    return {
        "month": "2025-01",
        "town": "ang mo kio",
        "flat_type": "4-room",
        "flat_model": "improved",
        "storey_range": "07 to 09",
        "floor_area_sqm": 90,
        "lease_commence_date": 2025
    }


def create_function_declarations(valid_values: dict):
    return [
        {
            "name": "call_prediction_api",
            "description": "Call the resale price prediction API with extracted parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {
                        "type": "string",
                        "description": "Month in YYYY-MM format. Default: 2025-01"
                    },
                    "town": {
                        "type": "string",
                        "enum": valid_values["towns"],
                        "description": f"Singapore town/estate name. Valid options: {', '.join(valid_values['towns'])}"
                    },
                    "flat_type": {
                        "type": "string",
                        "enum": valid_values["flat_types"],
                        "description": f"Type of HDB flat. Valid options: {', '.join(valid_values['flat_types'])}. Default: 4-room"
                    },
                    "flat_model": {
                        "type": "string",
                        "enum": valid_values["flat_models"],
                        "description": f"HDB flat model/design type. Valid options: {', '.join(valid_values['flat_models'])}. Default: improved"
                    },
                    "storey_range": {
                        "type": "string",
                        "enum": valid_values["storey_ranges"],
                        "description": f"Floor level range. Valid options: {', '.join(valid_values['storey_ranges'])}. Default: 07 to 09"
                    },
                    "floor_area_sqm": {
                        "type": "integer",
                        "description": f"Floor area in square meters. Must be between {valid_values['min_area']} and {valid_values['max_area']}. Default: 90"
                    },
                    "lease_commence_date": {
                        "type": "integer",
                        "description": "Year when lease commenced. Must be between 1960 and 2025. Default: 2025"
                    }
                },
                "required": ["town"]
            }
        },
        {
            "name": "call_analysis_api",
            "description": "Call the SQL analysis API to query historical data and trends",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The natural language query to analyze"
                    }
                },
                "required": ["query"]
            }
        }
    ]