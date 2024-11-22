import torch.nn as nn

class MLPGeo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, amount=4):
        super().__init__()
        self.mlps = nn.ModuleList([nn.Sequential(*(self.layer(input_size, hidden_size, output_size))) for _ in range(amount)])


    def layer(self, input_size, hidden_size, output_size):
        return [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)]
    
    def forward(self, x, idx):
        z = self.mlps[idx](x)

        return z