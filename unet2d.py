import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import DataParallel

def conv3d_layer_output_size(layer, input_dim):
    k_size=layer.kernel_size
    pad = layer.padding
    dil = layer.dilation
    stride = layer.stride

    D_out = math.floor((input_dim[1]+2*pad[0]-dil[0]*(k_size[0]-1)-1)/stride[0] +1)
    H_out = math.floor((input_dim[2]+2*pad[1]-dil[1]*(k_size[1]-1)-1)/stride[1] +1)
    W_out = math.floor((input_dim[3]+2*pad[2]-dil[2]*(k_size[2]-1)-1)/stride[2] +1)

    return (layer.out_channels, D_out, H_out, W_out)

class DownLayer(nn.Module):
    def __init__(self, in_size, num_filters, padding, batch_norm, dropout, activation):
        super().__init__()

        block = []

        block.append(nn.Conv2d(in_size, num_filters, kernel_size=3, stride=1, padding=1))
        if batch_norm:
            #block.append(nn.BatchNorm3d(num_filters))
            block.append(nn.GroupNorm(num_groups=4, num_channels = num_filters))

        block.append(nn.PReLU())

        if dropout:
            block.append(nn.Dropout2d(0.5))

        block.append(nn.Conv2d(num_filters, num_filters*2, kernel_size=3, stride=1, padding=1))
        if batch_norm:
            #block.append(nn.BatchNorm3d(num_filters*2))
            block.append(nn.GroupNorm(num_groups=4, num_channels = num_filters*2))

        block.append(nn.PReLU())

        if dropout:
            block.append(nn.Dropout2d(0.5))

        self.block = nn.Sequential(*block)

    def forward(self,x):
        #x = x.cuda()
        return self.block(x)

class UpLayer(nn.Module):
    def __init__(self, num_in_filters, up_mode,  padding, batch_norm, dropout, activation):
        super().__init__()
        self.upmode = up_mode
        if up_mode=='upconv':
            self.upconv = nn.ConvTranspose2d(num_in_filters, num_in_filters, kernel_size=2, stride=2)
        else:
            self.upconv = nn.Conv2d(num_in_filters, num_in_filters, kernel_size=1)

        num_out_filters=num_in_filters//2
        block=[]
        block.append(nn.Conv2d(num_in_filters+num_out_filters, num_out_filters, kernel_size=3, stride=1, padding=1))
        if batch_norm:
            #block.append(nn.BatchNorm3d(num_out_filters))
            block.append(nn.GroupNorm(num_groups=4, num_channels = num_out_filters))

        block.append(nn.PReLU())

        if dropout:
            block.append(nn.Dropout2d(0.5))

        block.append(nn.Conv2d(num_out_filters, num_out_filters, kernel_size=3, stride=1, padding=1))
        if batch_norm:
            #block.append(nn.BatchNorm3d(num_out_filters))
            block.append(nn.GroupNorm(num_groups=4, num_channels = num_out_filters))

        block.append(nn.PReLU())

        if dropout:
            block.append(nn.Dropout2d(0.5))

        self.block = nn.Sequential(*block)



    def forward(self, x, encoded):
        if self.upmode == 'upsample':
            x = torch.nn.functional.interpolate(x, mode='bilinear', scale_factor=2, align_corners=True)

        up = self.upconv(x)
        out = torch.cat([up, encoded],1)

        return self.block(out)


class Unet2d(nn.Module):
    def __init__(self, in_cnels, depth=4, n_classes=1, n_base_filters=32, padding=False, batch_norm=True, up_mode='upconv', activation='relu', final_activation=False):
        super().__init__()
        assert up_mode in ('upconv', 'upsample')
        assert activation in ('relu', 'elu')
        self.padding = padding
        self.depth = depth
        self.n_classes = n_classes
        self.final_activation = final_activation
        dropout = True
        #self.out_dim = input_dim[0]
        #in_channels=input_dim[0]
        in_channels = in_cnels

        self.pool = nn.MaxPool2d(2)

        # Encodeur
        self.down_path = nn.ModuleList()
        for i in range(depth):
            block = DownLayer(in_channels, n_base_filters*(2**i), padding, batch_norm, dropout, activation)
            self.down_path.append(block)
            # self.out_dim = conv3d_layer_output_size(block[0], self.out_dim)

            in_channels = n_base_filters*(2**i)*2

        # Decodeur
        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth-1)):
            block = UpLayer(n_base_filters*(2**(i+2)), up_mode, padding, batch_norm, dropout, activation)
            self.up_path.append(block)

        # Final 1x1x1 convolution
        output_size=n_classes
        if n_classes==2:
            output_size = 1

        self.last = nn.Conv2d(n_base_filters*2, output_size, kernel_size=1)   
        #self.last = CoordConv2d(n_base_filters*2, output_size, kernel_size=1)
        
    def forward(self,x):
        #print(x.size())
        blocks_out= []
        
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i!= len(self.down_path)-1:
                blocks_out.append(x)
                
                x = self.pool(x)
        #print("End encoder ====================================")
        
        for i, up in enumerate(self.up_path):
            x = up(x, blocks_out[self.depth-i-2])
            
        #print("End decoder ====================================")
        
        if self.final_activation:
            if self.n_classes==2:
                return nn.Sigmoid()(self.last(x))
            return nn.LogSoftmax(dim=1)(self.last(x))
        else:
            return self.last(x)

