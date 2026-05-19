"""
Convolutional Neural Network (CNN):

==================================================================================
Sample Code and Explanation:
==================================================================================

'''
class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B,16,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,16,14,14]

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # [B,32,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,32,7,7]
        )
        
        self.classifier = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
'''

==================================================================================
To explain the features part:
==================================================================================

1. Input is a grayscale image of size 28 x 28.
2. kernel_size = 3 and padding = 1 are common choices.
3. MaxPool2d(2) means 2 x 2 pooling, and there are 2 pooling layers. (normal to have  2 x 2)
4. In Conv2d(1, 16, ...), 1 means grayscale input channel, 16 is user-defined output channels.
5. Output becomes [B,16,28,28]:
   B = batch size, 16 = feature maps, 28 x 28 = image size.
6. After MaxPool2d(2) Downsample, 28 x 28 becomes 14 x 14, through a floor rule, depends on kernel, stride, padding
7. In Conv2d(16, 32, ...), 16 comes from previous output, and 32 is commonly chosen as double (16 x 2).
8. Before Linear, output [B,32,7,7] is flattened to 32 x 7 x 7.
9. Linear maps 32 x 7 x 7 features to num_classes outputs.

==================================================================================
Remarks:
==================================================================================

- The number of feature maps (channels) is a user-defined design choice.
    -> Smaller values such as 8 or 16 give a lighter model.
    -> Larger values such as 32 or 64 can improve learning capacity, but require more computation.

- kernel size
    -> if 3, then model looks at 3 x 3 patch each time
    -> large kernel need more parameters

- stride:
    -> how many pixels it moves every step
    -> move more, get output faster
    -> by default 1 if not specify

- padding:
    -> extra border around the image

"""



import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Model(nn.Module):
    
    def __init__(self, channel_size : int ,input_size : tuple , hidden_size : int , num_output : int , kernel_size : int , stride : int, padding : int , pool_size : int = 2):
        
        super().__init__()

        self.features = nn.Sequential(
            
            nn.Conv2d( channel_size , hidden_size, kernel_size = kernel_size, stride = stride, padding = padding ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pool_size, stride = stride),

            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size = pool_size, stride = stride),

        )
        
        flat_dim = self._get_flatten_dim(input_shape=(channel_size , * input_size ))
        
        self.classifier = nn.Linear(flat_dim, num_output)

   
    def _get_flatten_dim(self, input_shape):
        
        
        with torch.no_grad():
        
            x = torch.zeros(1, * input_shape)
            x = self.features(x)
            return x.flatten(1).shape[1]            



    def forward(self, x):
        
        x = self.features(x)
        x = x.flatten(1)

        return self.classifier(x)
    


