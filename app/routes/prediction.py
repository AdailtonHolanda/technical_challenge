import pandas as pd
from typing import Dict
from fastapi import APIRouter, HTTPException
from ..models.schemas import HomeData
from ..utils.model_loader import model, required_features_all, demographics_df, bonus_features_ordered

# Create an APIRouter to group the endpoints
router = APIRouter()


@router.post("/predict_all")
async def predict_price_all_features(home_data: HomeData):
    """
    Predicts the price of a house using all input data
    and adding demographic data in the backend.
    """
    try:
        input_df = pd.DataFrame([home_data.model_dump()])
        merged_df = pd.merge(input_df, demographics_df, on='zipcode', how='left')

        if merged_df.isnull().values.any():
            raise HTTPException(status_code=400, detail="Zipcode not found in demographic data.")

        # Reorder the columns to match the model features
        features_to_predict = merged_df[required_features_all]

        # Make the prediction
        prediction = model.predict(features_to_predict).tolist()[0]

        return {
            "prediction": prediction,
            "metadata": {
                "model_version": "1.0",
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_bonus")
async def predict_price_required_features(data: Dict):
    """
    Predicts the price of a house using only the required features,
    as specified by the bonus part of the challenge.
    """
    try:
        #input_df = pd.DataFrame([data])
        all_features_df = pd.DataFrame(columns=required_features_all)
        all_features_df.loc[0] = 0

        for feature, value in data.items():
            if feature in all_features_df.columns:
                all_features_df[feature] = value

        prediction = model.predict(all_features_df).tolist()[0]
 
        return {
            "prediction": prediction,
            "metadata": {
                "model_version": "1.0",
                "features_used": bonus_features_ordered
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))