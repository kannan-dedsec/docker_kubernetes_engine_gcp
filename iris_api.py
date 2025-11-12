from fastapi import FastAPI, Request
from pydantic import BaseModel
from google.cloud import logging as cloud_logging
from google.cloud.logging.handlers import CloudLoggingHandler
import joblib
import numpy as np
import logging

# Initialize FastAPI
app = FastAPI()

# Set up Google Cloud Logging
client = cloud_logging.Client()
handler = CloudLoggingHandler(client)
cloud_logger = logging.getLogger("iris-predictor")
cloud_logger.setLevel(logging.INFO)
cloud_logger.addHandler(handler)

# Load the model once at startup (avoid reloading for every request)
MODEL_PATH = "model/model.joblib"
try:
    model = joblib.load(MODEL_PATH)
    cloud_logger.info("Model loaded successfully from %s", MODEL_PATH)
except Exception as e:
    cloud_logger.error(f"Error loading model: {e}")
    raise e

# Define input schema
class InputVector(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health():
    return {"status": "ok", "message": "Service is up and running"}

@app.post("/predict")
async def predict(input: InputVector, request: Request):
    try:
        data = np.array([[input.sepal_length, input.sepal_width,
                          input.petal_length, input.petal_width]])
        preds = model.predict(data)
        prediction = preds.tolist()

        cloud_logger.info({
            "prediction": prediction,
            "input": input.dict(),
            "client": request.client.host
        })

        return {"prediction": prediction}
    except Exception as e:
        cloud_logger.exception("Prediction failed")
        return {"error": str(e)}

# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
