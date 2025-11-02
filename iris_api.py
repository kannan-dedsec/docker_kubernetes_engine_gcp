from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np

class InputVector(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

MODEL_PATH = "model/model.joblib"

@app.get("/")
def read_root():
    return {"message": "Welcome to IRIS FastAPI!"}

@app.post("/predict")
def predict(input: InputVector):
    model = joblib.load(MODEL_PATH)
    data = np.array([[input.sepal_length, input.sepal_width,
                      input.petal_length, input.petal_width]])
    preds = model.predict(data)
    return {"prediction": preds.tolist()}
