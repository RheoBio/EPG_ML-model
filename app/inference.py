import torch
from model import EPGTransformerCNN  ## load pytorch model

class Model:
    def __init__(self, input_size, hidden_size, output_size, filename):# load model using filename
        """
        Load the model
        """
        self.model = EPGTransformerCNN(input_size, hidden_size, output_size)
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()

    def run(self, data):
        with torch.no_grad():
            input_tensor = data[0]  # Assuming data[0] is the first argument 'data'
            x_cwt = data[1]  # Assuming data[1] is the second argument 'data_x_cwt'
            out = self.model(input_tensor, x_cwt)
            return out

    """
    def run(self, data):

        Prediction

        with torch.no_grad():
            input_tensor = torch.tensor(data[0])  # Assuming data[0] is the first argument 'data'
            x_cwt = torch.tensor(data[1])  # Assuming data[1] is the second argument 'data_x_cwt'
            out = self.model(input_tensor, x_cwt)
            return out
    """
    """
    def run(self, data, x_cwt):

        Prediction

        with torch.no_grad():
            input_tensor = torch.tensor(data)  # Assuming data is a tensor or array-like object
            out = self.model(input_tensor, x_cwt)
            return out
    """