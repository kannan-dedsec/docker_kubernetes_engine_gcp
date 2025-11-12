from fastapi import FastAPI
import joblib
import numpy as np


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome its working !! "}

@app.post("/predict")
def predict():
    return {"prediction": "setosa"}
