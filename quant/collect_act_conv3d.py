import argparse
import datetime
import numpy as np
import os
import re
import torch
import pickle
from pathlib import Path
from functools import partial
from spconv.pytorch.conv import SparseConv3d, SubMConv3d
from spconv.pytorch.modules import SparseModule
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

from tools.eval_utils import eval_utils
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import common_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# global pytorch_outputs
# pytorch_outputs = {}
# weight_scale = {}

class QuantConv3d(SparseModule):
    """Pytorch Quantization"""
    def __init__(self, spconv3d):
        super().__init__()
        self.spconv3d = spconv3d
        # quantize w
        w_desc = QuantDescriptor(
            num_bits=8, 
            axis=(0)
        )
        w_quant = TensorQuantizer(w_desc)
        oc, kh, kw, kd, ic = self.spconv3d.weight.data.shape
        w = self.spconv3d.weight.data.permute(0, 4, 1, 2, 3).contiguous().view(oc, -1)
        w = w_quant(w)
        self.spconv3d.weight.data = w.view(oc, ic, kh, kw, kd).permute(0, 2, 3, 4, 1).contiguous()

        # quantize act
        act_desc = QuantDescriptor(
            num_bits=8,
            # unsigned=True
        )
        self.act_quant = TensorQuantizer(act_desc)

        return

    def forward(self, x):
        features = x.features
        features = self.act_quant(features)
        x = x.replace_feature(features)
        x = self.spconv3d(x)
        return x


class SQConv3d(SparseModule):
    """Pytorch Quantization"""
    def __init__(self, spconv3d):
        super().__init__()
        self.spconv3d = spconv3d
        # quantize w
        w_desc = QuantDescriptor(
            num_bits=8, 
            axis=(0)
        )
        self.w_quant = TensorQuantizer(w_desc)

        # quantize act
        act_desc = QuantDescriptor(
            num_bits=8,
            # unsigned=True
        )
        self.act_quant = TensorQuantizer(act_desc)
        self.original_weight = self.spconv3d.weight.data.clone()

    def forward(self, x):
        features = x.features
        features_max = features.abs().max()
        weight_max = self.spconv3d.weight.data.abs().max()
        scale = torch.sqrt(features_max/weight_max)
        if scale == 0:
            scale = 1
        print(f'features_max: {features_max}\t| weight_max: {weight_max}\t| scale: {scale}')

        features /= scale
        features = self.act_quant(features)
        x = x.replace_feature(features)

        oc, kh, kw, kd, ic = self.spconv3d.weight.data.shape
        w = self.spconv3d.weight.data.permute(0, 4, 1, 2, 3).contiguous().view(oc, -1)
        w *= scale
        w = self.w_quant(w)

        self.spconv3d.weight.data = w.view(oc, ic, kh, kw, kd).permute(0, 2, 3, 4, 1).contiguous()
        
        x = self.spconv3d(x)
        self.spconv3d.weight.data = self.original_weight.clone()
        return x


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
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
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
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    torch.manual_seed(4)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def sq_conv3d(model, module_dict, curr_path, alpha, act_num_bits, weight_num_bits):

    for name, module in model.named_children():
        # print(module)
        path = f"{curr_path}.{name}" if curr_path else name
        sq_conv3d(module, module_dict, path, alpha, act_num_bits, weight_num_bits)
        if isinstance(module, (SubMConv3d, SparseConv3d)) and path != 'backbone_3d.conv_input.0':
        # if isinstance(module, SubMConv3d):
            # print(module)
            # replace layer with Pytorch Quantization
            # model._modules[name] = QuantConv3d(spconv3d=module)
            model._modules[name] = SQConv3d(spconv3d=module)
            # replace layer with SQ Quantization
            # model._modules[name] = QuantConv3d(spconv3d=module, scaling_factor=0.5)

    return


def register_collect_input_hook(model):

        def forward_hook(module, input, name):
            global pytorch_outputs, weight_scale
            tmp = input[0].features.detach().cpu().abs().flatten()
            tmp_weight = module.weight.data.detach().cpu().abs().flatten()
            values, _ = torch.topk(tmp, 5)
            weight_values, _ = torch.topk(tmp_weight, 5)
            pytorch_outputs[name]= values.tolist()
            weight_scale[name] = weight_values.tolist()
            
            
        for name, module in model.named_modules():
            if isinstance(module, (SparseConv3d, SubMConv3d)):
                module.register_forward_pre_hook(partial(forward_hook, name=name))


def main() -> None:
    global pytorch_outputs, weight_scale
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

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)
    model.cuda()

    sq_conv3d(model, module_dict={}, curr_path="", alpha=0.5, act_num_bits=8, weight_num_bits=8)

    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )

    # register_collect_input_hook(model)

    # with torch.no_grad():
    #     for batch_dict in test_loader:
    #         load_data_to_gpu(batch_dict)
    #         model(batch_dict)
    #         break
    
    # # Save the input activations
    # with open('pytorch_outputs.pkl', 'wb') as f:
    #     pickle.dump([pytorch_outputs, weight_scale], f)
    
    # print(pytorch_outputs)
    # print(weight_scale)
    # print('Done!')

    return


if __name__ == '__main__':
    main()