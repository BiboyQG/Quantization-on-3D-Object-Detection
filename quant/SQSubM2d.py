import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter


class SQSubM2d(nn.Module):
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

        if isinstance(self.padding, (list)):
            self.padding = self.padding[0]
        if isinstance(self.dilation, (list)):
            self.dilation = self.dilation[0]
        if isinstance(self.kernel_size, (list)):
            self.kernel_size = self.kernel_size[0]
        if isinstance(self.stride, (list)):
            self.stride = self.stride[0]

        h_out = math.floor((h_in + 2*self.padding - self.dilation*(self.kernel_size-1)-1)/self.stride+1)
        w_out = math.floor((w_in + 2*self.padding - self.dilation*(self.kernel_size-1)-1)/self.stride+1)
        
        # x: [bs ksize num_sliding]

        print('input x:', input.shape)

        x = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

        print('x after unfold:', x.shape)

        bs = input.shape[0]
        ksize = self.in_channels*self.kernel_size*self.kernel_size
        num_sliding = x.shape[2]

        assert x.shape[1] == ksize

        # x: [bs*num_sliding ksize]
        x = torch.transpose(x, 1, 2).reshape(-1, ksize)

        print('x after reshape:', x.shape)

        weight_flat = self.weight.data.clone().view(self.out_channels, ksize)

        tensor_x = x.abs().detach()
        tensor_weight = weight_flat.abs().detach()

        act_scale = torch.max(tensor_x, dim=0)[0]
        weight_scale = torch.max(tensor_weight, dim=0)[0]

        scale = act_scale**self.scaling_factor/weight_scale**(1-self.scaling_factor)
        scale[scale==0] = 1

        x /= scale
        weight_flat = weight_flat * scale

        print('x before quantizer:', x.shape)

        x = self._input_quantizer(x)
        weight_flat = self._weight_quantizer(weight_flat)

        print('x after quantizer:', x.shape)

        x = x.reshape(bs, x.shape[0], -1).transpose(1, 2)

        x = torch.nn.functional.fold(x, (h_in, w_in), kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

        print('x after fold:', x.shape)

        x = x.permute(0, 2, 3, 1)

        # x = x.reshape(bs, num_sliding, self.out_channels)
        # x = torch.transpose(x, 1, 2)
        # x = x.reshape(bs, self.out_channels, h_out, w_out)

        weight = weight_flat.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).permute(0, 2, 3, 1).contiguous()
        return weight, x