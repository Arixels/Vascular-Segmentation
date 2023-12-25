# %%

import torch
import torch.nn as nn


def crop_to_target_size(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]

    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

def double_conv(input, output):
    # print(input)
    conv = nn.Sequential(
        nn.Conv2d(input, output, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(output, output, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
                
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(1, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)
        
        self.up_trans1 = nn.ConvTranspose2d(in_channels=1024,
                                            out_channels=512, 
                                            kernel_size=2,
                                            stride=2)
        
        self.up_conv1 = double_conv(1024, 512)
        
        self.up_trans2 = nn.ConvTranspose2d(in_channels=512,
                                            out_channels=256, 
                                            kernel_size=2,
                                            stride=2)
        
        self.up_conv2 = double_conv(512, 256)
        
        self.up_trans3 = nn.ConvTranspose2d(in_channels=256,
                                            out_channels=128, 
                                            kernel_size=2,
                                            stride=2)
        
        self.up_conv3 = double_conv(256, 128)
        
        self.up_trans4 = nn.ConvTranspose2d(in_channels=128,
                                            out_channels=64, 
                                            kernel_size=2,
                                            stride=2)
        
        self.up_conv4 = double_conv(128, 64)
        
        self.out = nn.Conv2d(in_channels=64,
                             out_channels=2,    # increase number of channels based on number of objects to segment
                             kernel_size=1)
        
    
    def forward(self, image):
        # bs, c, h, w
        # encoder
        # print(image)
        x1 = self.down_conv1(image)
        print(x1.size())
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv5(x8)
        
        # decoder
        x = self.up_trans1(x9)
        x7_cropped = crop_to_target_size(x7, x)
        x = self.up_conv1(torch.cat([x, x7_cropped], 1))
        
        x = self.up_trans2(x)
        
        x5_cropped = crop_to_target_size(x5, x)
        x = self.up_conv2(torch.cat([x, x5_cropped], 1))
        
        x = self.up_trans3(x)
        
        x3_cropped = crop_to_target_size(x3, x)
        x = self.up_conv3(torch.cat([x, x3_cropped], 1))
        
        
        x = self.up_trans4(x)
        
        x1_cropped = crop_to_target_size(x1, x)
        x = self.up_conv4(torch.cat([x, x1_cropped], 1))
        
        x = self.out(x)
        
        print(x.size())
        return x 
        
        # return "hello"
        
# 
# %%
# if __name__ == "__main_(_":
image = torch.rand((1, 1, 572, 572))
# print(image)
model = UNet()
model.forward(image)
# %%
