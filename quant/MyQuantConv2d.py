import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter


class MyQuantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, input_quantizer=None, weight_quantizer=None, device='cuda', scaling_factor=None):
        super(MyQuantConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self._input_quantizer = input_quantizer
        self._weight_quantizer = weight_quantizer
        self.scaling_factor = scaling_factor

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)).to(device)

    def forward(self, input):
        if self.scaling_factor is None:
            raise ValueError("Please specify the scaling_factor parameter!")
        h_in, w_in = input.shape[2:]

        h_out = math.floor((h_in + 2*self.padding - self.dilation*(self.kernel_size-1)-1)/self.stride+1)
        w_out = math.floor((w_in + 2*self.padding - self.dilation*(self.kernel_size-1)-1)/self.stride+1)

        # x: [bs ksize num_sliding]
        x = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

        bs = input.shape[0]
        ksize = self.in_channels*self.kernel_size*self.kernel_size
        num_sliding = x.shape[2]

        assert x.shape[1] == ksize

        # x: [bs*num_sliding ksize]
        x = torch.transpose(x, 1, 2).reshape(-1, ksize)

        weight_flat = self.weight.view(self.out_channels, ksize)

        tensor_x = x.abs().detach()
        tensor_weight = weight_flat.abs().detach()

        act_scale = torch.max(tensor_x, dim=0)[0]
        weight_scale = torch.max(tensor_weight, dim=0)[0]

        scale = act_scale**self.scaling_factor/weight_scale**(1-self.scaling_factor)
        scale[scale==0] = 1

        x /= scale
        weight_flat = weight_flat * scale

        x = self._input_quantizer(x)
        weight_flat = self._weight_quantizer(weight_flat)
        
        x = torch.mm(x, weight_flat.t())

        x = x.reshape(bs, num_sliding, self.out_channels)
        x = torch.transpose(x, 1, 2)
        x = x.reshape(bs, self.out_channels, h_out, w_out)
        return x