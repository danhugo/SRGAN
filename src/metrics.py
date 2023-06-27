import torch
import torch.nn.functional as F
import constants
import numpy as np

def gauss1D(window_size, sigma):
    center = window_size // 2
    gauss = torch.Tensor([np.exp(-(x - center)**2 / (2*(sigma**2))) for x in range(window_size)])
    gauss = gauss/gauss.sum()
    return gauss
        
def create_window(window_size, sigma, channels: int = 3):
    window1d = gauss1D(window_size, sigma).unsqueeze(1)
    window2d = torch.mm(window1d, window1d.t())
    window2d = window2d.repeat(channels, 1, 1, 1)
    return window2d

def rgb_to_ycbcr(image: torch.Tensor, only_use_y_channel: bool = True) -> torch.Tensor:
    """Convert RGB Image to YCbCr Image
    
    Args:
    - image (Tensor): Tensor image shape (B, 3, H, W)
    - only_use_y_channel (bool): whether or not extract image with only Y channel.

    Returns:
    - Tensor image: shape (B, 1, H, W) if only_use_y_channel is True and (B, 3, H, W) the other way.
    """

    if not isinstance(image, torch.Tensor) or image.size(-3) != 3:
        raise ValueError("Invalid format of image, should be Tensor(B, 3, H, W)")
    image = image.to(constants.DEVICE)
    if only_use_y_channel:
        weight = torch.tensor([[65.481], [128.533], [24.966]]).to(constants.DEVICE)
        image = torch.matmul(image.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0],
                               [128.553, -74.203, -93.786],
                               [24.966, 112.0, -18.214]]).to(constants.DEVICE)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(constants.DEVICE)
        image = torch.matmul(image.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
    
    image /= 255.
    return image

def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int, sigma: float, channels: int, batch_average: bool = True) -> torch.Tensor:
    """Caculate SSIM of 2 images.
    
    Returns:
    - Tensor: value of SSIM, which is (B,) if batch_average is not True and scalar if True.
    """

    # to device
    window = create_window(window_size, sigma, channels).to(constants.DEVICE)
    img1 = img1.to(constants.DEVICE)
    img2 = img2.to(constants.DEVICE)

    c1 = (0.01 * constants.PIXEL_VALUE_RANGE)**2
    c2 = (0.03 * constants.PIXEL_VALUE_RANGE)**2
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channels) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channels) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channels) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + c1)*(2*sigma12 + c2))/((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))
    if batch_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=(1,2,3))

class Metrics():
    def __init__(
            self,
            extract_y_channel: bool = True) -> None:
        """ Caculate PSNR and SSIM metrics.
        - extract_y_channel: whether or not extract y channel in YCrCb format
            then PSNR and SSIM will be computed on only y channel images.
        """
        self.extract_y_channel = extract_y_channel

    def extractYchannel(self):
        self.lowres = rgb_to_ycbcr(self.lowres)
        self.highres = rgb_to_ycbcr(self.highres)
        
    def psnr(self, img1: torch.Tensor, img2: torch.Tensor):
        """"""
        img1 = img1.to(constants.DEVICE)
        img2 = img2.to(constants.DEVICE)
        
        rmse = torch.sqrt(F.mse_loss(img1, img2))
        psnr = 20 * torch.log10(constants.PIXEL_VALUE_RANGE/ (rmse + 1e-10))
        return psnr

    def ssim(self, img1: torch.Tensor, img2: torch.Tensor):
        """"""
        return _ssim(img1, img2, window_size=11, sigma=0.15, channels=img1.size(-3))
