from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from model import StockPredictor
import os
from fastapi.middleware.cors import CORSMiddleware
from model import TickerDataError

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    ticker: str = 'AAPL'
    days_to_predict: int = 5

class PredictionResponse(BaseModel):
    ticker: str
    predictions: List[dict]
    message: str
    historical_data: List[dict]

predictors = {}

@app.on_event("startup")
async def startup_event():
    pass

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    ticker = request.ticker.upper()
    if ticker not in predictors:
        predictors[ticker] = StockPredictor(ticker=ticker)
        try:
            predictors[ticker].load_trained_model()
            predictors[ticker].load_scaler()
            predictors[ticker].load_data()
        except TickerDataError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            try:
                predictors[ticker].load_data()
                x_train, y_train, _, _ = predictors[ticker].preprocess_data()
                predictors[ticker].build_model()
                predictors[ticker].train_model(x_train, y_train)
                predictors[ticker].save_model()
                predictors[ticker].save_scaler()
            except TickerDataError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as train_e:
                raise HTTPException(status_code=500, detail=f"Failed to train model for {ticker}: {train_e}")
    
    try:
        predictors[ticker].ticker = ticker
        predictors[ticker].load_data()

        future_preds_df = predictors[ticker].predict_future(days_to_predict=request.days_to_predict)
        
        look_back = predictors[ticker].look_back
        historical_df = predictors[ticker].df['Close'].tail(look_back + request.days_to_predict * 2).reset_index()
        historical_list = historical_df.to_dict(orient='records')
        for item in historical_list:
            item['Date'] = item['Date'].strftime('%Y-%m-%d')

        predictions_list = future_preds_df.reset_index().to_dict(orient='records')
        for item in predictions_list:
            item['Date'] = item['Date'].strftime('%Y-%m-%d')

        return PredictionResponse(
            ticker=ticker,
            predictions=predictions_list,
            historical_data=historical_list,
            message=f"Prediction for {ticker} for {request.days_to_predict} days."
        )
    except TickerDataError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed for {ticker}: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded_count": len(predictors)}
