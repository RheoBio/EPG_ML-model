from pydantic import BaseModel
from fastapi import FastAPI
import logging
import os
import inference
import uvicorn

# load model
model = inference.Model("model_file.h5 pytorch model here")

app = FastAPI()
logger = logging.getLogger("AI")

# SET DEPLOYMENT MODE =================
deployment_mode = None
try:
    deployment_mode = os.environ.get('deployment_mode')  # 'local' or 'AWS'
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>> deployment mode: {} <<<<<<<<<<<<<<<<<<<<<<<<<<'.format(deployment_mode))
except Exception as e:
    logger.error(e)


# DEFINE TYPES
class InputData(BaseModel):
    data: list[float]


class OutputData(BaseModel):
    data: list[float]


# DEFINE ROUTES
@app.get("/")
def home():
    return {"welcome": "to AI algorithms"}


@app.post("/classify_noise_signal")
def classify_noise_signal(x: InputData):
    out = model.run(x)
    out = list(out)
    return OutputData(data=out)


if __name__ == '__main__':
    if deployment_mode == 'local':
        host = 'localhost'
        port = 8003
    else:
        host = '0.0.0.0'
        port = 8000
    uvicorn.run("__main__:app", host=host, port=port, reload_dirs="./", reload=True, log_level=logging.DEBUG)
