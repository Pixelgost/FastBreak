import torch.onnx
import onnx
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Since we have 8 vectors of 4 elements each, input dimension will be 8 * 4 = 32
        self.input_dim = 8 * 5
        self.hidden_dim1 = 80
        self.hidden_dim2 = 40
        self.output_dim = 1 

        # Define the layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, self.output_dim)

    def forward(self, x):
        x = x.view(-1, 8 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x