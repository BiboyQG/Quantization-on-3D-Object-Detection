import argparse
import datetime
import numpy as np
import os
import re
import torch
import torch.nn as nn
from pathlib import Path
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization.tensor_quant import QuantDescriptor
from QConvNd import QConvNd
from smoothquant import SQConv2d, SQConvT2d
from spconv.pytorch.conv import SubMConv3d, SparseConv3d
from tools.eval_utils import eval_utils
from tqdm import tqdm


# sh scripts/dist_test.sh 3 --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml --ckpt ../weights/cbgs_voxel0075_centerpoint_nds_6648.pth

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

no_list = [
    'dense_head.rpn_heads.0.conv_box.conv_reg.3',
    'dense_head.rpn_heads.0.conv_box.conv_height.3',
    'dense_head.rpn_heads.0.conv_box.conv_size.3',
    'dense_head.rpn_heads.0.conv_box.conv_angle.3',
    'dense_head.rpn_heads.0.conv_box.conv_velo.3',
    'dense_head.rpn_heads.0.conv_cls.3',
    'dense_head.rpn_heads.1.conv_box.conv_reg.3',
    'dense_head.rpn_heads.1.conv_box.conv_height.3',
    'dense_head.rpn_heads.1.conv_box.conv_size.3',
    'dense_head.rpn_heads.1.conv_box.conv_angle.3',
    'dense_head.rpn_heads.1.conv_box.conv_velo.3',
    'dense_head.rpn_heads.1.conv_cls.3',
    'dense_head.rpn_heads.2.conv_box.conv_reg.3',
    'dense_head.rpn_heads.2.conv_box.conv_height.3',
    'dense_head.rpn_heads.2.conv_box.conv_size.3',
    'dense_head.rpn_heads.2.conv_box.conv_angle.3',
    'dense_head.rpn_heads.2.conv_box.conv_velo.3',
    'dense_head.rpn_heads.2.conv_cls.3',
    'dense_head.rpn_heads.3.conv_box.conv_reg.3',
    'dense_head.rpn_heads.3.conv_box.conv_height.3',
    'dense_head.rpn_heads.3.conv_box.conv_size.3',
    'dense_head.rpn_heads.3.conv_box.conv_angle.3',
    'dense_head.rpn_heads.3.conv_box.conv_velo.3',
    'dense_head.rpn_heads.3.conv_cls.3',
    'dense_head.rpn_heads.4.conv_box.conv_reg.3',
    'dense_head.rpn_heads.4.conv_box.conv_height.3',
    'dense_head.rpn_heads.4.conv_box.conv_size.3',
    'dense_head.rpn_heads.4.conv_box.conv_angle.3',
    'dense_head.rpn_heads.4.conv_box.conv_velo.3',
    'dense_head.rpn_heads.4.conv_cls.3',
    'dense_head.rpn_heads.5.conv_box.conv_reg.3',
    'dense_head.rpn_heads.5.conv_box.conv_height.3',
    'dense_head.rpn_heads.5.conv_box.conv_size.3',
    'dense_head.rpn_heads.5.conv_box.conv_angle.3',
    'dense_head.rpn_heads.5.conv_box.conv_velo.3',
    'dense_head.rpn_heads.5.conv_cls.3',
]


def q_conv3d(model, module_dict, curr_path, w_bits, act_bits, cw):
    for name, module in model.named_children():
        path = f"{curr_path}.{name}" if curr_path else name
        q_conv3d(module, module_dict, path, w_bits, act_bits, cw)
        if isinstance(module, (SubMConv3d, SparseConv3d)) and \
            path != 'backbone_3d.conv_input.0':
            # replace layer with standard quantization
            model._modules[name] = QConvNd(
                module=module,
                w_bits=w_bits,
                act_bits=act_bits,
                cw=cw,
            )
            # replace layer with SQ Quantization(currently unable to perform SQ)
            # model._modules[name] = SQConv3d(module=module, scaling_factor=0.5)
    return


def smoothquant_layer(
        nn_instance,
        quant_mudule,
        scaling_factor,
        w_bits,
        act_bits,
) -> None:
    if scaling_factor is None:
        raise ValueError("Please specify the scaling_factor parameter!")
    if act_bits is None or w_bits is None:
        raise ValueError("Please specify the num_bits parameter!")
    quant_instance = quant_mudule.__new__(quant_mudule)
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
) -> None:
    if alpha is None:
        raise ValueError("Please specify the scaling_factor parameter!")

    if act_bits is None or w_bits is None:
        raise ValueError("Please specify the num_bits parameter!")

    for name, module in model.named_children():
        # Update the path for each submodule
        path = f"{curr_path}.{name}" if curr_path else name
        # Recursively process each submodule
        smoothquant(module, module_dict, path, alpha, w_bits, act_bits, src, tgt)

        if isinstance(module, src) and path not in no_list:
            model._modules[name] = smoothquant_layer(
                module,
                tgt,
                alpha,
                w_bits,
                act_bits,
            )
    return


def pytorch_quant_layer(nn_instance, quant_mudule, w_bits, act_bits):
    quant_instance = quant_mudule.__new__(quant_mudule)
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
    if act_bits is None or w_bits is None:
        raise ValueError("Please specify the num_bits parameter!")

    for name, module in model.named_children():
        # Update the path for each submodule
        path = f"{curr_path}.{name}" if curr_path else name
        # Recursively process each submodule
        pytorch_quant(module, module_dict, path, w_bits, act_bits, src, tgt)

        if isinstance(module, src) and path not in no_list:
            model._modules[name] = pytorch_quant_layer(module, tgt, w_bits, act_bits)
    return


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


def dynamic_quant(
        model,
        w_bits: int,
        act_bits: int,
        sq: bool,
        alpha: float,
) -> None:
    q_conv3d(
        model,
        module_dict={},
        curr_path="",
        w_bits=w_bits,
        act_bits=act_bits,
        cw=sq,
    )
    if sq:
        smoothquant(
            model,
            module_dict={},
            curr_path="",
            alpha=alpha,
            w_bits=w_bits,
            act_bits=act_bits,
            src=(nn.Conv2d),
            tgt=SQConv2d,
        )
        # smoothquant(
        #     model,
        #     module_dict={},
        #     curr_path="",
        #     alpha=alpha,
        #     w_bits=w_bits,
        #     act_bits=act_bits,
        #     src=(nn.ConvTranspose2d),
        #     tgt=SQConvT2d,
        # )
    else:
        pytorch_quant(
            model,
            module_dict={},
            curr_path="",
            w_bits=w_bits,
            act_bits=act_bits,
            src=(nn.Conv2d),
            tgt=quant_nn.Conv2d,
        )
        # pytorch_quant(
        #     model,
        #     module_dict={},
        #     curr_path="",
        #     w_bits=w_bits,
        #     act_bits=act_bits,
        #     src=(nn.ConvTranspose2d),
        #     tgt=quant_nn.ConvTranspose2d,
        # )

    return


def static_quant(
        model,
        w_bits: int,
        act_bits: int,
        sq: bool,
        alpha: float,
        test_loader,
        n_batches,
) -> None:
    q_conv3d(
        model,
        module_dict={},
        curr_path="",
        w_bits=w_bits,
        act_bits=act_bits,
        cw=sq,
    )
    if sq:
        smoothquant(
            model,
            module_dict={},
            curr_path="",
            alpha=alpha,
            w_bits=w_bits,
            act_bits=act_bits,
            src=(nn.Conv2d),
            tgt=SQConv2d,
        )
        # smoothquant(
        #     model,
        #     module_dict={},
        #     curr_path="",
        #     alpha=alpha,
        #     w_bits=w_bits,
        #     act_bits=act_bits,
        #     src=(nn.ConvTranspose2d),
        #     tgt=SQConvT2d,
        # )
    else:
        pytorch_quant(
            model,
            module_dict={},
            curr_path="",
            w_bits=w_bits,
            act_bits=act_bits,
            src=(nn.Conv2d),
            tgt=quant_nn.Conv2d,
        )
        # pytorch_quant(
        #     model,
        #     module_dict={},
        #     curr_path="",
        #     w_bits=w_bits,
        #     act_bits=act_bits,
        #     src=(nn.ConvTranspose2d),
        #     tgt=quant_nn.ConvTranspose2d,
        # )

    collect_stats(model, test_loader, n_batches)

    compute_amax(model, device, method='entropy')

    return


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    # NEED THIS I DONNO WHY
    # args.local_rank = int(os.environ['LOCAL_RANK'])
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    # ========== SET SEED ==========
    seed = 4
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    # ==============================

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main() -> None:
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=test_set,
    )
    model.load_params_from_file(
        filename=args.ckpt,
        logger=logger,
        to_cpu=False,
        pre_trained_path=args.pretrained_model,
    )
    model.cuda()

    # ========== dynamic ==========
    dynamic_quant(model, w_bits=8, act_bits=8, sq=True, alpha=0.5)
    # =============================

    # ========== static ==========
    # static_quant(
    #     model,
    #     w_bits=8,
    #     act_bits=8,
    #     sq=True,
    #     alpha=0.5,
    #     test_loader=test_loader,
    #     n_batches=200,
    # )
    # ============================

    logger.info(model)

    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )

    return


if __name__ == '__main__':
    main()
