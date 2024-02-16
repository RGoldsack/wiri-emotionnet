import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class CustomModel(nn.Module):
    def __init__(self, n_inputs, hidden_nodes, dropout, runtimeOptions):
        super(CustomModel, self).__init__()
        self.lstm = nn.LSTM(n_inputs, hidden_nodes, batch_first=True, 
                            stateful=runtimeOptions[3])
        self.dense1 = nn.Linear(hidden_nodes, n_inputs)
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(n_inputs, n_inputs)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_y1 = self.layer_y(n_inputs, 512, 256, dropout, 1)
    
    def layer_y(self, input_size, outputDenseSize_L1, outputDenseSize_L2, dropout, size):
        layers = nn.Sequential(
            nn.Linear(input_size, outputDenseSize_L1),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(outputDenseSize_L1, outputDenseSize_L2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(outputDenseSize_L2, size),
        )
        return layers
    
    def forward(self, x, lengths):
        # Pack the padded sequences
        x_packed = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM and other operations on packed sequences
        x_packed, _ = self.lstm(x_packed)
        
        # Unpack sequences
        x, _ = rnn_utils.pad_packed_sequence(x_packed, batch_first=True)
        x, _ = self.lstm(x)
        x = F.leaky_relu(self.dense1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense2(x))
        x = self.dropout2(x)
        y = self.layer_y1(x)

        return y
