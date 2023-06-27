from torchvision.models import vgg19
import torch.nn as nn
import constants

class VGGLoss(nn.Module):
    def __init__(self) -> None:
        """Caculate MSE loss feature map after 4th conv layer before 5th pooling layer"""
        super(VGGLoss, self).__init__()
        self.model = vgg19(weights='DEFAULT').features[:35].to(constants.DEVICE)
    
    def forward(self, x):
        return self.model(x)
