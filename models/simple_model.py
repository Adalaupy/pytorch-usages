
"""
Simple Feed-Forward Network (MLP)

- Tabular data
- regression
- binary/multi-class classification

"""



import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple_Model(nn.Module):
    
    def __init__(self, input_size : int, hidden_size : int, num_output : int, num_hidden_layer: int, dropout : float):        

        super().__init__()

        self.layers = nn.ModuleList()
        self.num_output = num_output


        for i in range( num_hidden_layer + 2 ):
            
            # First layer
            if i == 0:
                                
                self.layers.append( nn.Linear(input_size ,hidden_size ) )

            # Output layer
            if i == num_hidden_layer + 1:
                
                self.layers.append(nn.Linear(hidden_size, num_output))

            # Hidden layer
            else:
                
                self.layers.append(nn.Linear(hidden_size, hidden_size))

        
        
        self.dropout = nn.Dropout(dropout)



        
    def forward(self, x):
        
        for layer in self.layers[:-1]:
            
            x = F.relu(layer(x))

            x = self.dropout(x)


        # for classification:
        if self.num_output == 2:
            
            x = F.sigmoid( self.layers[-1](x) )
        

        # for regression:
        else:

            x = self.layers[-1](x)


        return x