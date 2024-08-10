import torch
from pcdet.models import load_data_to_gpu
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor
from quant import QConvNd
from tqdm import tqdm


# ================================ Quant Conv3d ================================
def q_conv3d(
        model,
        module_dict,
        curr_path,
        w_bits,
        act_bits,
        cw,
        src,
        no_list,
) -> None:
    for name, module in model.named_children():
        path = f"{curr_path}.{name}" if curr_path else name
        q_conv3d(
            module,
            module_dict,
            path,
            w_bits,
            act_bits,
            cw,
            src,
            no_list,
        )
        if isinstance(module, src) and path not in no_list:
            model._modules[name] = QConvNd(
                module=module,
                w_bits=w_bits,
                act_bits=act_bits,
                cw=cw,
            )

    return
# ==============================================================================


# ================================ Smooth Quant ================================
def smoothquant_layer(
        nn_instance,
        quant_module,
        scaling_factor,
        w_bits,
        act_bits,
) -> None:
    if scaling_factor is None:
        raise ValueError("Please specify the scaling_factor parameter!")
    if act_bits is None or w_bits is None:
        raise ValueError("Please specify the num_bits parameter!")
    quant_instance = quant_module.__new__(quant_module)
    for k, val in vars(nn_instance).items():
        if isinstance(val, tuple):
            val = val[0]
        setattr(quant_instance, k, val)
    w_desc = QuantDescriptor(
        num_bits=w_bits, 
        axis=(0),
    )
    act_desc = QuantDescriptor(
        num_bits=act_bits,
        # unsigned=True
    )
    w_quant = TensorQuantizer(w_desc)
    act_quant = TensorQuantizer(act_desc)
    quant_instance._weight_quantizer = w_quant
    quant_instance._input_quantizer = act_quant
    quant_instance.scaling_factor = scaling_factor
    return quant_instance


def smoothquant(
        model,
        module_dict,
        curr_path,
        alpha,
        w_bits,
        act_bits,
        src,
        tgt,
        no_list,
) -> None:
    for name, module in model.named_children():
        # Update the path for each submodule
        path = f"{curr_path}.{name}" if curr_path else name
        # Recursively process each submodule
        smoothquant(
            module,
            module_dict,
            path,
            alpha,
            w_bits,
            act_bits,
            src,
            tgt,
            no_list,
        )

        if isinstance(module, src) and path not in no_list:
            model._modules[name] = smoothquant_layer(
                module,
                tgt,
                alpha,
                w_bits,
                act_bits,
            )
    return
# ==============================================================================


# =============================== PyTorch Quant ================================
def pytorch_quant_layer(nn_instance, quant_module, w_bits, act_bits):
    quant_instance = quant_module.__new__(quant_module)
    for k, val in vars(nn_instance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        # Return two instances of QuantDescriptor;
        # self.__class__ is the class of quant_instance, E.g.: QuantConv2d
        # quant_desc_input, quant_desc_weight =
        # quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        quant_desc_weight = QuantDescriptor(
            num_bits=w_bits, 
            axis=(0), 
        )
        quant_desc_input = QuantDescriptor(
            num_bits=act_bits,
            # unsigned=True
        )
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance


def pytorch_quant(
        model,
        module_dict,
        curr_path,
        w_bits,
        act_bits,
        src,
        tgt,
) -> None:
    for name, module in model.named_children():
        # Update the path for each submodule
        path = f"{curr_path}.{name}" if curr_path else name
        # Recursively process each submodule
        pytorch_quant(module, module_dict, path, w_bits, act_bits, src, tgt)

        if isinstance(module, src):
            model._modules[name] = pytorch_quant_layer(module, tgt, w_bits, act_bits)
    return
# ==============================================================================


# ================================ static quant ================================
def collect_stats(model, data_loader, n_batches=200):
    model.eval()
    for name, module in model.named_modules():
        if name.endswith('_quantizer') or name.endswith('_quant'):
            module.enable_calib()
            module.disable_quant()

    with torch.no_grad():
        for i, batch_dict in enumerate(
            tqdm(data_loader, desc='calibration', total=n_batches+1)
        ):
            load_data_to_gpu(batch_dict)
            model(batch_dict)
            if i > n_batches:
                break

    for name, module in model.named_modules():
        if name.endswith('_quantizer') or name.endswith('_quant'):
            module.disable_calib()
            module.enable_quant()
    return


def compute_amax(model, device, **kwargs):
    for _, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax(strict=False)
                else:
                    module.load_calib_amax(**kwargs)
                module._amax = module._amax.to(device)
    return
# ==============================================================================
