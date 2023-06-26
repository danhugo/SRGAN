from torchvision.models import vgg19
import torch.nn as nn
import torch
import constants

class VGGLoss(nn.Module):
    def __init__(self) -> None:
        """Caculate MSE loss feature map after 4th conv layer before 5th pooling layer"""
        super(VGGLoss, self).__init__()
        
        # phi 5,4: fmaps of 4th convolutional layer before 5th pooling layer (after activation)
        # corresponding to features[35]
        self.model = vgg19(weights='DEFAULT').features[:18].to(constants.DEVICE)
    
    def forward(self, x):
        return self.model(x)

        
    