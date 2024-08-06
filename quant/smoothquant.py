import torch
import torch.nn as nn
import torch.nn.functional as F


class SQConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            input_quantizer=None,
            weight_quantizer=None,
            device='cuda',
            scaling_factor=None,
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self._weight_quantizer = weight_quantizer
        self._input_quantizer = input_quantizer
        self.scaling_factor = scaling_factor

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        ).to(device)
        self.bias = nn.Parameter(torch.Tensor(out_channels)).to(device)

        return

    def forward(self, x):
        if self.scaling_factor is None:
            raise ValueError("Please specify the scaling_factor parameter!")

        if isinstance(self.padding, (list)):
            self.padding = self.padding[0]
        if isinstance(self.dilation, (list)):
            self.dilation = self.dilation[0]
        if isinstance(self.kernel_size, (list)):
            self.kernel_size = self.kernel_size[0]
        if isinstance(self.stride, (list)):
            self.stride = self.stride[0]

        ickhkw = self.in_channels*self.kernel_size*self.kernel_size

        w = self.weight.view(self.out_channels, ickhkw)

        bs, _, h_in, w_in = x.shape

        # [bs ic*kh*kw num_sliding]
        x = F.unfold(
            input=x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )

        # [bs*num_sliding ic*kh*kw]
        x = x.permute(0, 2, 1).reshape(-1, ickhkw)

        w_tensor = w.abs().detach()
        x_tensor = x.abs().detach()

        w_scale = torch.max(w_tensor, dim=0)[0]
        act_scale = torch.max(x_tensor, dim=0)[0]

        scale = act_scale**self.scaling_factor/w_scale**(1-self.scaling_factor)
        scale[scale==0] = 1

        w = w*scale
        x /= scale

        w = self._weight_quantizer(w)
        x = self._input_quantizer(x)

        # [bs*num_sliding oc]
        x @= w.T

        # [bs oc num_sliding]
        x = x.view(bs, -1, self.out_channels).permute(0, 2, 1)

        h_out = (h_in+2*self.padding-self.dilation*(self.kernel_size-1)-1)//self.stride+1
        w_out = (w_in+2*self.padding-self.dilation*(self.kernel_size-1)-1)//self.stride+1

        x = x.view(bs, self.out_channels, h_out, w_out)

        if self.bias is not None:
            b = self.bias.view(1, self.out_channels, 1, 1)
            x += b

        return x


class SQConv1d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            weight_quantizer=None,
            input_quantizer=None,
            device="cuda",
            scaling_factor=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self._weight_quantizer = weight_quantizer
        self._input_quantizer = input_quantizer
        self.scaling_factor = scaling_factor

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size)
        ).to(device)
        self.bias = nn.Parameter(torch.Tensor(out_channels)).to(device)

        return

    def forward(self, x):
        if self.scaling_factor is None:
            raise ValueError("Please specify the scaling_factor parameter!")

        w = self.weight.view(self.out_channels, -1)

        bs, ic, len = x.shape
        x = x.unsqueeze(dim=2)
        x = F.unfold(
            input=x,
            kernel_size=(1, self.kernel_size),
            dilation=(1, self.dilation),
            padding=(0, self.padding),
            stride=(1, self.stride),
        )
        # [b*len ic*ks]
        x = x.permute(0, 2, 1).reshape(-1, ic*self.kernel_size)

        w_tensor = w.abs().detach()
        act_tensor = x.abs().detach()
        w_scale = torch.max(w_tensor, dim=0)[0]
        act_scale = torch.max(act_tensor, dim=0)[0]

        scale = act_scale**self.scaling_factor/w_scale**(1-self.scaling_factor)
        scale[scale==0] = 1

        x /= scale
        w = w*scale

        w = self._weight_quantizer(w)
        x = self._input_quantizer(x)
        # [bs*out_len, oc]
        x @= w.T

        out_len = (len+2*self.padding-self.dilation*(self.kernel_size-1)-1)//self.stride+1

        # [bs, oc, out_len]
        x = x.view(bs, out_len, self.out_channels).permute(0, 2, 1)

        if self.bias is not None:
            x += self.bias.view(1, self.out_channels, 1)
        
        return x


class SQConvT2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            dilation=1,
            input_quantizer=None,
            weight_quantizer=None,
            device='cuda',
            scaling_factor=None,
        ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self._weight_quantizer = weight_quantizer
        self._input_quantizer = input_quantizer
        self.scaling_factor = scaling_factor

        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels, kernel_size, kernel_size)
        ).to(device)
        self.bias = nn.Parameter(torch.Tensor(out_channels)).to(device)

        return

    def forward(self, x):
        if self.scaling_factor is None:
            raise ValueError("Please specify the scaling_factor parameter!")

        if isinstance(self.padding, (list)):
            self.padding = self.padding[0]
        if isinstance(self.dilation, (list)):
            self.dilation = self.dilation[0]
        if isinstance(self.kernel_size, (list)):
            self.kernel_size = self.kernel_size[0]
        if isinstance(self.stride, (list)):
            self.stride = self.stride[0]

        bs, ic, ih, iw = x.shape

        # [oc*kh*kw ic]
        w = self.weight.view(self.in_channels, -1).T
        # [bs*ih*iw, ic]
        x = x.view(bs, ic, ih*iw).permute(0, 2, 1).view(-1, ic)

        w_tensor = w.abs().detach()
        x_tensor = x.abs().detach()

        w_scale = torch.max(w_tensor, dim=0)[0]
        act_scale = torch.max(x_tensor, dim=0)[0]

        scale = act_scale**self.scaling_factor/w_scale**(1-self.scaling_factor)
        scale[scale==0] = 1

        w = w*scale
        x /= scale

        w = self._weight_quantizer(w)
        x = self._input_quantizer(x)

        # [bs*ih*iw oc*kh*kw]
        x @= w.T

        # [bs ih*iw oc*kh*kw] -> [bs oc*kh*kw ih*iw]
        x = x.view(bs, ih*iw, -1).permute(0, 2, 1)

        h_out = (ih-1)*self.stride-2*self.padding+self.dilation*(self.kernel_size-1)+self.output_padding+1
        w_out = (iw-1)*self.stride-2*self.padding+self.dilation*(self.kernel_size-1)+self.output_padding+1

        x = F.fold(
            input=x,
            output_size=(h_out, w_out),
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )

        if self.bias is not None:
            b = self.bias.view(1, self.out_channels, 1, 1)
            x += b

        return x


class SQLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            weight_quantizer=None,
            input_quantizer=None,
            device="cuda",
            scaling_factor=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._weight_quantizer = weight_quantizer
        self._input_quantizer = input_quantizer
        self.scaling_factor = scaling_factor

        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features)
        ).to(device)
        self.bias = nn.Parameter(torch.Tensor(out_features)).to(device)

        return

    def forward(self, x):
        _, bs, _ = x.shape
        # [seq_len bs ic] -> [seq_len*bs ic]
        x = x.view(-1, self.in_features)
        w_tensor = self.weight.abs().detach()
        x_tensor = x.abs().detach()

        w_scale = torch.max(w_tensor, dim=0)[0]
        act_scale = torch.max(x_tensor, dim=0)[0]

        scale = act_scale**self.scaling_factor/w_scale**(1-self.scaling_factor)
        scale[scale==0] = 1

        w = self.weight*scale
        x /= scale

        w = self._weight_quantizer(w)
        x = self._input_quantizer(x)
        x @= w.T

        if self.bias is not None:
            x += self.bias.view(1, self.out_features)

        x = x.view(-1, bs, self.out_features)

        return x
