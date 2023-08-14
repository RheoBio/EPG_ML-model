from pydantic import BaseModel
from fastapi import FastAPI
import logging
import os
import inference
import uvicorn
import numpy as np
import torch
import torchvision.transforms as transforms
import pywt

# DEFINE TYPES
class InputData(BaseModel):
    sample_data: list[float]
    sample_rate: float


class OutputData(BaseModel):
    data_classification: list  #0-1
    msg : str


# Define the specific input_size, hidden_size, and output_size
size = 700
input_size = size
hidden_size = 300
output_size = size


# load model
model = inference.Model(input_size, hidden_size, output_size, "./model_ep85_sts_cpu.pt")

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


def cwt_transform(data):
    wavelet = 'morl'  # 使用Morlet小波
    scales = np.arange(1, 129)  # 定義尺度範圍

    cwt_coeffs = []
    for sample in data:
        coeffs, _ = pywt.cwt(sample, scales, wavelet)
        cwt_coeffs.append(coeffs)

    cwt_coeffs = np.array(cwt_coeffs)
    return cwt_coeffs


def cwt_stadardization(data_cwt):
  data_cwt = torch.tensor(data_cwt, dtype=torch.float32)
  # Calculate mean and standard deviation from training data
  mean_x_cwt = data_cwt.mean(axis=(0, 2), keepdim=True)
  std_x_cwt = data_cwt.std(axis=(0, 2), keepdim=True)
  result = (data_cwt - mean_x_cwt) / std_x_cwt
  return result

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
    processed_data = cut_list(x.sample_data,700, 50)
    data = torch.tensor(processed_data, dtype=torch.float32)
    data_x_cwt = cwt_transform(processed_data)
    data_x_cwt = cwt_stadardization(data_x_cwt)
    out = model.run((data, data_x_cwt))
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