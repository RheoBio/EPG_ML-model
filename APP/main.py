from pydantic import BaseModel
from fastapi import FastAPI
import logging
import os
import inference
import uvicorn
import numpy as np
import torch
import torchvision.transforms as transforms

# DEFINE TYPES
class InputData(BaseModel):
    processed_rawdata: list[float]


class OutputData(BaseModel):
    data_classification: np.ndarray

# load model
model = inference.Model("./model_CNN_0.88.pt")

#initialize a FastAPI instance
app = FastAPI()
logger = logging.getLogger("AI")

# SET DEPLOYMENT MODE =================
deployment_mode = None
try:
    deployment_mode = os.environ.get('deployment_mode')  # 'local' or 'AWS'
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>> deployment mode: {} <<<<<<<<<<<<<<<<<<<<<<<<<<'.format(deployment_mode))
except Exception as e:
    logger.error(e)

# DEFINE ROUTES
@app.get("/")
def get_root():
    return {"welcome": "to noise&signal detection algorithms"}


@app.post("/classify_noise_signal")

#
def classify_noise_signal(input_data: InputData):
    data = torch.tensor(input_data, dtype=torch.float32)
    out = model.run(data)
    out = out.numpy().tolist()
    return OutputData(data_classification=out)


if __name__ == '__main__':
    if deployment_mode == 'local':
        host = 'localhost'
        port = 8003
    else:
        host = '0.0.0.0'
        port = 8000
    uvicorn.run("__main__:app", host=host, port=port, reload_dirs="./", reload=True, log_level=logging.DEBUG)