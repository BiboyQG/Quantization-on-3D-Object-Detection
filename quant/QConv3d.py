from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor
from spconv.pytorch.modules import SparseModule


class QConv3d(SparseModule):
    """Pytorch Quantization"""
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
        oc, kh, kw, kd, ic = self.module.weight.data.shape
        self.module.weight.data = self.module.weight.data.permute(0, 4, 1, 2, 3).contiguous().view(oc, -1)
        self.module.weight.data = self.w_quant(self.module.weight.data)
        self.module.weight.data = self.module.weight.data.view(oc, ic, kh, kw, kd).permute(0, 2, 3, 4, 1).contiguous()

        features = x.features
        features = self.act_quant(features)
        x = x.replace_feature(features)

        x = self.module(x)

        self.module.weight.data = self.w.clone()

        return x
