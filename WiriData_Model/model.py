# Local Modules

# Installed Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
# Default Modules
import warnings

warnings.filterwarnings("ignore")

class CustomModel(nn.Module):
    def __init__(self, n_inputs, hidden_nodes, dropout):
        super(CustomModel, self).__init__()
        self.gru = nn.GRU(n_inputs, hidden_nodes, batch_first=True)
        
        # Decreasing size linear layers
        self.dense1 = nn.Linear(hidden_nodes, hidden_nodes // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_nodes // 2, hidden_nodes // 4)
        self.dropout2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(hidden_nodes // 4, hidden_nodes // 8)
        self.dropout3 = nn.Dropout(dropout)
        
        # Increasing size linear layers
        self.dense4 = nn.Linear(hidden_nodes // 8, hidden_nodes // 4)
        self.dropout4 = nn.Dropout(dropout)
        self.dense5 = nn.Linear(hidden_nodes // 4, hidden_nodes // 2)
        self.dropout5 = nn.Dropout(dropout)
        self.dense6 = nn.Linear(hidden_nodes // 2, hidden_nodes)
        self.dropout6 = nn.Dropout(dropout)
        
        # Final layer to match desired output size
        self.final_dense = nn.Linear(hidden_nodes, 1)
    
    def forward(self, x):
        # Directly pass the input through the GRU layer
        x, _ = self.gru(x)
        print(f"x_1: {x}")
        
        # Pass through decreasing size layers
        x = F.leaky_relu(self.dense1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense3(x))
        x = self.dropout3(x)
        
        # Pass through increasing size layers
        x = F.leaky_relu(self.dense4(x))
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense5(x))
        x = self.dropout5(x)
        x = F.leaky_relu(self.dense6(x))
        x = self.dropout6(x)
        
        # Final output
        x = self.final_dense(x)
            
        return x

