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
    sample_data: list[float]
    sample_rate: float


class OutputData(BaseModel):
    data_classification: list  #0-1
    msg : str

# load model
model = inference.Model("./model_CNN_0.88.pt")

#initialize a FastAPI instance
app = FastAPI()
logger = logging.getLogger("AI")

def cut_list(sample, size, overlap):
    data = []
    start = 0
    while start + size <= len(sample):
      data.append(sample[start : start + size])
      start += size - overlap

    else :
      last_chunk = sample[len(sample)-size:len(sample)]
      data.append(last_chunk)

    return np.array(data)

# SET DEPLOYMENT MODE =================
deployment_mode = None
try:
    deployment_mode = os.environ.get('deployment_mode')  # 'local' or 'AWS'
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>> deployment mode: {} <<<<<<<<<<<<<<<<<<<<<<<<<<'.format(deployment_mode))
except Exception as e:
    logger.error(e)

# DEFINE ROUTES
#@app.get("/")
#async def get_root():
#    return {"welcome": "to noise&signal detection algorithms"}

@app.post("//classification")
def classify_noise_signal(x: InputData):
    if x.sample_rate == 100:
        pass
    else:
        return OutputData(data_classification = [], msg = "bad sample rate")
    processed_data = cut_list(x.sample_data,440, 50)
    input_data = torch.tensor(processed_data, dtype=torch.float32)
    data = torch.tensor(input_data)
    out = model.run(data)
    out = out.numpy().tolist()
    return OutputData(data_classification=out, msg = "")


if __name__ == '__main__':
    if deployment_mode == 'local':
        host = 'localhost'
        port = 8003
    else:
        host = '0.0.0.0'
        port = 8000
    uvicorn.run("__main__:app", host=host, port=port, reload_dirs="./", reload=True, log_level=logging.DEBUG)