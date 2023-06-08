from model import EPGCNN  ## load pytorch model
import torch

class Model:
    def __init__(self, filename):# load model using filename
        """
        Load the model
        """
        self.model = EPGCNN()
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()

    def run(self, data):
        """
        Prediction
        """
        with torch.no_grad():
            input_tensor = torch.tensor(data)  # Assuming data is a tensor or array-like object
            out = self.model(input_tensor)
            return out