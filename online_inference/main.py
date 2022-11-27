from fastapi import FastAPI
import uvicorn
import pickle
import os
from pydantic import BaseModel, conlist
import logging
import pandas as pd
from typing import List, Union

logger = logging.getLogger("Server")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

class Disease(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=13, max_items=13)]
    features: List[str]

class ModelResponse(BaseModel):
    disease: int

app = FastAPI()

model = None
transformer = None

def load_object(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"object {path} is loaded")
    return obj
    
def make_predict(data, features):
    global model
    df = pd.DataFrame(data=data, columns=features)
    X = transformer.transform(df)
    y = model.predict(X)
    logger.info(f"predicted: {y}")
    return [ModelResponse(disease=y)]

@app.on_event("startup")
def load_model():
    global model
    global transformer
    
    model_path = os.getenv("PATH_TO_MODEL")
    transformer_path = os.getenv("PATH_TO_TRANSFORMER")

    for elem in [model_path, transformer_path]:
        if elem is None:
            logger.error(f"{elem} is None")
            raise RuntimeError(f"{elem} is None")

    model = load_object(model_path)
    transformer = load_object(transformer_path)

@app.get("/")
def main():
    return "It is entry point of our predictor"

@app.get("/health")
def health():
    if model is None or transformer is None:
        logger.info("Model and transformer is not ready")
        return False
    else:
        logger.info("Model and transformer is ready")
        return True

@app.post("/predict/", response_model=List[ModelResponse])
async def predict(request: Disease):
    logger.info(request)
    return make_predict(request.data, request.features)



if __name__ == "__main__":
    logger.info("Server started")
    uvicorn.run("main:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
