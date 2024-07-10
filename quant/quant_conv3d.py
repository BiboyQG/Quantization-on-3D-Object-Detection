import argparse
import datetime
import numpy as np
import os
import re
import spconv
import torch
import torch.nn as nn
import unfoldNd
from pathlib import Path
from spconv.pytorch.conv import SparseConv3d, SubMConv3d
from spconv.pytorch.modules import SparseModule
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

from tools.eval_utils import eval_utils
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import common_utils

# sh scripts/dist_test.sh 3 --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml --ckpt ../weights/cbgs_voxel0075_centerpoint_nds_6648.pth

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pytorch_outputs = {}
sq_outputs = {}


class QuantConv3d(SparseModule):
    """Pytorch Quantization"""
    def __init__(self, spconv3d, act_bits, w_bits):
        super().__init__()
        self.spconv3d = spconv3d

        # quantize w
        w_desc = QuantDescriptor(
            num_bits=w_bits,
            axis=(0)
        )
        w_quant = TensorQuantizer(w_desc)
        oc, kh, kw, kd, ic = self.spconv3d.weight.data.shape
        w = self.spconv3d.weight.data.permute(0, 4, 1, 2, 3).contiguous().view(oc, -1)
        w = w_quant(w)
        self.spconv3d.weight.data = w.view(oc, ic, kh, kw, kd).permute(0, 2, 3, 4, 1).contiguous()

        # quantize act
        act_desc = QuantDescriptor(
            num_bits=act_bits,
            # unsigned=True
        )
        self.act_quant = TensorQuantizer(act_desc)

        self.spconv3d = spconv3d
        return

    def forward(self, x):
        features = x.features
        features = self.act_quant(features)
        x = x.replace_feature(features)
        x = self.spconv3d(x)
        return x


class SQConv3d(SparseModule):
    """SmoothQuant Quantization"""
    def __init__(self, spconv3d, scaling_factor):
        super().__init__()
        self.spconv3d = spconv3d
        self.scaling_factor = scaling_factor
        self.oc, self.kd, self.kh, self.kw, self.ic = self.spconv3d.weight.shape
        self.dh, self.dw, self.dd = self.spconv3d.stride
        
        return

    def forward(self, x):
        # SQ act
        x = x.dense()
        x = x.cpu()
        print("x", x.shape)

        b, c, d, h, ww = x.shape

        x = x.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)\
            .unfold(4, self.kd, self.dd)
        print("unfold", x.shape)

        # [b, xxx, ic*kh*kw*kd]
        x = x.contiguous().view(b, c, -1, self.kh, self.kw, self.kd)\
            .permute(0, 2, 1, 3, 4, 5).flatten(2)
        # [b*xxx, ic*kh*kw*kd]
        x = x.view(-1, x.shape[-1])
        print("tx x", x.shape)

        act_scale = torch.max(x.abs_(), dim=0)[0]
        print("act scale", act_scale.shape)

        # SQ w
        # [oc, ic*kh*kw,kd]
        w = self.spconv3d.weight.data.permute(0, 4, 1, 2, 3)\
            .contiguous().view(self.oc, -1).transpose(0, 1)
        print("tx w", w.shape)

        w_scale = torch.max(w.abs(), dim=1)[0]
        w_scale = w_scale.cpu()
        print("w scale", w_scale.shape)

        scale = act_scale**self.scaling_factor/w_scale**(1-self.scaling_factor)
        print("scale", scale.shape)

        scale[scale==0] = 1
        # print(scale)

        x /= scale

        scale = scale.to(device)
        w = w.T
        w *= scale
        w = w.T
        self.spconv3d.weight.data = w.contiguous()\
            .view(self.oc, self.ic, self.kd, self.kh, self.kw).permute(0, 2, 3, 4, 1)

        x = x.view(b, -1, x.shape[-1]).permute(0, 2, 1)
        print("tx x", x.shape)

        # ====== start folding back here ======
        out_h = int((h + 2 * 1 - self.kh) / self.dh + 1)
        out_w = int((ww + 2 * 1 - self.kw) / self.dw + 1)
        out_d = int((d + 2 * 1 - self.kd) / self.dd + 1)
        print('out_h:', out_h)
        print('out_w:', out_w)
        print('out_d:', out_d)

        # self.fold = nn.Fold(
        #     output_size=[out_h, out_w, out_d],
        #     kernel_size=[self.kd, self.kh, self.kw],
        #     padding=1,
        #     stride=self.spconv3d.stride,
        # )
        # x = self.fold(x)

        foldnd = unfoldNd.FoldNd(
            (out_h, out_w, out_d),
            kernel_size=(self.kh, self.kw, self.kd),
            dilation=1,
            padding=1,
            stride=self.spconv3d.stride,
        )
        x = foldnd(x)
        # exit()
        print("fold x", x.shape)

        # ====================================

        x = x.to(device)

        x = spconv.SparseConvTensor.from_dense(x)

        x = self.spconv3d(x)
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

    np.random.seed(1024)

    torch.manual_seed(4)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def q_conv3d(model, module_dict, curr_path, w_bits, act_bits):
    for name, module in model.named_children():
        # print(module)
        path = f"{curr_path}.{name}" if curr_path else name
        q_conv3d(module, module_dict, path, w_bits, act_bits)
        if isinstance(module, (SparseConv3d,)) and path != 'backbone_3d.conv_input.0':
            # print(module)
            # replace layer with Pytorch Quantization
            model._modules[name] = QuantConv3d(spconv3d=module, w_bits=w_bits, act_bits=act_bits)
            # replace layer with SQ Quantization (currently unable to perform SQ)
            # model._modules[name] = SQConv3d(spconv3d=module, scaling_factor=0.5)

    return


def pytorch_hook(module, input, output):
    pytorch_outputs[module] = output.detach()


def reg_hook(model):
    for name, module in model.named_modules():
        if isinstance(module, (QuantConv3d, SQConv3d)):
            module.register_forward_hook(pytorch_hook)


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

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)
    model.cuda()

    q_conv3d(model, module_dict={}, curr_path="", w_bits=8, act_bits=8)
    print(model)

    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )

    return


if __name__ == '__main__':
    main()