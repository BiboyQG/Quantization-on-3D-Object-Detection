from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor
from spconv.pytorch.modules import SparseModule


class QConvNd(SparseModule):
    """Quant Module for SubMConvNd & SparseConvNd"""
    def __init__(self, module, w_bits: int, act_bits: int, cw: bool):
        super().__init__()
        self.module = module
        self.w = self.module.weight.data.clone()

        # quantize w
        w_desc = QuantDescriptor(
            num_bits=w_bits,
            axis=(0)
        )
        self.w_quant = TensorQuantizer(w_desc)

        # quantize act
        if cw:
            act_desc = QuantDescriptor(
                num_bits=act_bits,
                axis=(1),
                # unsigned=True
            )
        else:
            act_desc = QuantDescriptor(
                num_bits=act_bits,
                # unsigned=True
            )
        self.act_quant = TensorQuantizer(act_desc)

        return

    def forward(self, x):
        oc = self.module.weight.data.shape[0]
        ic = self.module.weight.data.shape[-1]
        kdim = self.module.weight.data.shape[1:-1]
        dim = self.module.weight.data.dim()
        permute_dim = [0, dim-1] + list(range(1, dim-1))
        self.module.weight.data = self.module.weight.data.permute(permute_dim).contiguous().view(oc, -1)
        self.module.weight.data = self.w_quant(self.module.weight.data)
        permute_dim = [0] + list(range(2, dim)) + [1]
        shape = (oc, ic) + tuple(kdim)
        self.module.weight.data = self.module.weight.data.view(shape).permute(permute_dim).contiguous()

        features = x.features
        features = self.act_quant(features)
        x = x.replace_feature(features)

        x = self.module(x)

        self.module.weight.data = self.w.clone()

        return x
