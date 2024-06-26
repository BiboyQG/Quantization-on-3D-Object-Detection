from tools import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from functools import partial
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from tensorboardX import SummaryWriter

from tools.eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

from pytorch_quantization import quant_modules
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from absl import logging as quant_logging

from MyQuantConv2d import MyQuantConv2d

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
                                pre_trained_path=args.pretrained_model)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


# global_time = None
# exec_times = []
# module_names = []
# gpu_usage = []
global my_res, res, quant_res
my_res = {}
res = {}
quant_res = {}

def main():
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

    # test_set, test_loader, sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=args.batch_size,
    #     dist=dist_test, workers=args.workers, logger=logger, training=False
    # )

    # sample = next(iter(test_loader))
    # load_data_to_gpu(sample)

    # model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    # model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

    # model.cuda()

    global my_res, res

    def register_time_hooks(model):
        def forward_hook(module, input, output):
            global global_time, exec_times, module_names
            exec_times.append(time.time() - global_time)
            module_names.append(str(module.__class__))
            global_time = time.time()

        handles = []
        for module in model.modules():
            handle = module.register_forward_hook(forward_hook)
            handles.append(handle)
        return handles

    
    def log_time(model):
        global module_names, exec_times
        for module, t in zip(model.modules(), exec_times):
            logger.info(f"{module.__class__}: {t} seconds")

        module_time_pairs = list(zip(module_names, exec_times))

        sorted_module_time_pairs = sorted(module_time_pairs, key=lambda x: x[1], reverse=True)

        top_twenty = sorted_module_time_pairs[:20]
            
        for module, t in top_twenty:
            logger.info(f"{module}: {t} seconds")
    
    class MemoryUsageMonitor:
        global module_names, gpu_usage
        def __init__(self):
            self.reset()

        def reset(self):
            self.memory_allocated_before = 0
            self.memory_allocated_after = 0

        def pre_forward_hook(self, module, input):
            self.memory_allocated_before = torch.cuda.memory_allocated()

        def forward_hook(self, module, input, output):
            self.memory_allocated_after = torch.cuda.memory_allocated()
            module_names.append(str(module.__class__))
            gpu_usage.append(self.memory_allocated_after - self.memory_allocated_before)

    def register_gpu_hooks(model, monitor):
        for layer in model.modules():
            layer.register_forward_pre_hook(monitor.pre_forward_hook)
            layer.register_forward_hook(monitor.forward_hook)

    def log_gpu(model):
        global module_names, gpu_usage
        for module, g in zip(model.modules(), gpu_usage):
            logger.info(f"{module.__class__}: {g / 1024**2} MB")

        module_gpu_pairs = list(zip(module_names, gpu_usage))

        sorted_module_gpu_pairs = sorted(module_gpu_pairs, key=lambda x: x[1], reverse=True)

        top_twenty = sorted_module_gpu_pairs[:20]
            
        for module, t in top_twenty:
            logger.info(f"{module}: {t} MB")

    # eval_utils.eval_one_epoch(
    #     cfg, args, model, test_loader, epoch_id, logger, dist_test=False,
    #     result_dir=eval_output_dir
    # )

    def initialize_all(calib_method: str, num_bits: int = 16):
        """
        This method is used to initialize the default Descriptor for Conv2d activation quantization:
            1. intput QuantDescriptor: Max or Histogram
            2. calib_method -> ["max", "histogram"]
        :param calib_method: ["max", "histogram"]
        :return: no return value
        """
        quant_desc_input = QuantDescriptor(calib_method=calib_method, num_bits=num_bits, unsigned=True)
        quant_desc_weight = QuantDescriptor(num_bits=num_bits)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
        quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)
        quant_logging.set_verbosity(quant_logging.ERROR)

    def initialize(calib_method: str):
        """
        This method is used to initialize the default Descriptor for Conv2d activation quantization:
            1. intput QuantDescriptor: Max or Histogram
            2. calib_method -> ["max", "histogram"]
        :param calib_method: ["max", "histogram"]
        :return: no return value
        """
        quant_desc_input = QuantDescriptor(calib_method=calib_method)
        quant_desc_input_self = QuantDescriptor(num_bits=8, calib_method=calib_method, unsigned=True)
        # quant_desc_input_self_trans = QuantDescriptor(num_bits=16, calib_method=calib_method, unsigned=True)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        # quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input_self_trans)
        quant_logging.set_verbosity(quant_logging.ERROR)

    def initialize_unsigned_for_input_and_convtranspose(calib_method: str):
        """
        This method is used to initialize the default Descriptor for Conv2d activation quantization:
            1. intput QuantDescriptor: Max or Histogram
            2. calib_method -> ["max", "histogram"]
        :param calib_method: ["max", "histogram"]
        :return: no return value
        """
        quant_desc_input = QuantDescriptor(calib_method=calib_method)
        quant_desc_input_self = QuantDescriptor(num_bits=8, calib_method=calib_method, unsigned=True)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input_self)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input_self)
        quant_logging.set_verbosity(quant_logging.ERROR)

    def initialize_unsigned_for_input(calib_method: str):
        """
        This method is used to initialize the default Descriptor for Conv2d activation quantization:
            1. intput QuantDescriptor: Max or Histogram
            2. calib_method -> ["max", "histogram"]
        :param calib_method: ["max", "histogram"]
        :return: no return value
        """
        quant_desc_input = QuantDescriptor(calib_method=calib_method)
        quant_desc_input_self = QuantDescriptor(calib_method=calib_method, num_bits=8, unsigned=True)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input_self)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_logging.set_verbosity(quant_logging.ERROR)

    def initialize_weight(calib_method: str, num_bits: int = 16):
        """
        This method is used to initialize the default Descriptor for Conv2d activation quantization:
            1. intput QuantDescriptor: Max or Histogram
            2. calib_method -> ["max", "histogram"]
        :param calib_method: ["max", "histogram"]
        :return: no return value
        """
        quant_desc_input = QuantDescriptor(calib_method=calib_method, num_bits=16)
        quant_desc_input_self = QuantDescriptor(num_bits=16)
        quant_desc_weight = QuantDescriptor(num_bits=num_bits)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input_self)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_weight)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_weight)
        quant_logging.set_verbosity(quant_logging.ERROR)

    def initialize_input(calib_method: str, num_bits: int = 16):
        """
        This method is used to initialize the default Descriptor for Conv2d activation quantization:
            1. intput QuantDescriptor: Max or Histogram
            2. calib_method -> ["max", "histogram"]
        :param calib_method: ["max", "histogram"]
        :return: no return value
        """
        quant_desc_input = QuantDescriptor(calib_method=calib_method, num_bits=16)
        quant_desc_input_self = QuantDescriptor(num_bits=num_bits, calib_method=calib_method)
        quant_desc_weight = QuantDescriptor(num_bits=16)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input_self)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_weight)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_weight)
        quant_logging.set_verbosity(quant_logging.ERROR)

    def replace_to_quantization_model(model, ignore_layer=None):
        """
        Entry point of quantizing the model: this function is used to filter out the ignored layers and initialize the replace map.
        :param model: model returned from "prepare_model" function
        :param ignore_layer: layers that we don't want to be quantized
        :return: no return value
        """
        module_dict = {}
        for entry in quant_modules._DEFAULT_QUANT_MAP:
            module = getattr(entry.orig_mod, entry.mod_name)
            module_dict[id(module)] = entry.replace_mod
        torch_module_find_quant_module(model, module_dict, ignore_layer)

    def valid_replace_to_quantization_model(model, ignore_layer=None, scaling_factor=0.5, act_num_bits=None, weight_num_bits=None):
        """
        Entry point of quantizing the model: this function is used to filter out the ignored layers and initialize the replace map.
        :param model: model returned from "prepare_model" function
        :param ignore_layer: layers that we don't want to be quantized
        :return: no return value
        """
        module_dict = {}
        for entry in quant_modules._DEFAULT_QUANT_MAP:
            module = getattr(entry.orig_mod, entry.mod_name)
            module_dict[id(module)] = entry.replace_mod
        vali_original_head_torch_module_find_quant_module(model, module_dict, ignore_layer, current_path=None, scaling_factor=scaling_factor, act_num_bits=act_num_bits, weight_num_bits=weight_num_bits)
    
    def my_head_replace_to_quantization_model(model, ignore_layer=None, scaling_factor=None, act_num_bits=None, weight_num_bits=None):
        """
        Entry point of quantizing the model: this function is used to filter out the ignored layers and initialize the replace map.
        :param model: model returned from "prepare_model" function
        :param ignore_layer: layers that we don't want to be quantized
        :return: no return value
        """
        if scaling_factor is None:
            raise ValueError("Please specify the scaling_factor parameter!")
        if act_num_bits is None or weight_num_bits is None:
            raise ValueError("Please specify the num_bits parameter!")
        module_dict = {}
        for entry in quant_modules._DEFAULT_QUANT_MAP:
            module = getattr(entry.orig_mod, entry.mod_name)
            # if entry.mod_name == 'Conv2d':
            #     module_dict[id(module)] = MyQuantConv2d
            # else: 
            module_dict[id(module)] = entry.replace_mod
        test_head_torch_module_find_quant_module(model, module_dict, ignore_layer, current_path=None, scaling_factor=scaling_factor, act_num_bits=act_num_bits, weight_num_bits=weight_num_bits)

    def my_original_replace_to_quantization_model(model, ignore_layer=None, scaling_factor=None, act_num_bits=None, weight_num_bits=None):
        """
        Entry point of quantizing the model: this function is used to filter out the ignored layers and initialize the replace map.
        :param model: model returned from "prepare_model" function
        :param ignore_layer: layers that we don't want to be quantized
        :return: no return value
        """
        if scaling_factor is None:
            raise ValueError("Please specify the scaling_factor parameter!")
        if act_num_bits is None or weight_num_bits is None:
            raise ValueError("Please specify the num_bits parameter!")
        module_dict = {}
        for entry in quant_modules._DEFAULT_QUANT_MAP:
            module = getattr(entry.orig_mod, entry.mod_name)
            # if entry.mod_name == 'Conv2d':
            #     module_dict[id(module)] = MyQuantConv2d
            # else: 
            module_dict[id(module)] = entry.replace_mod
        test_original_torch_module_find_quant_module(model, module_dict, ignore_layer, current_path=None, scaling_factor=scaling_factor, act_num_bits=act_num_bits, weight_num_bits=weight_num_bits)

    def my_replace_to_quantization_model(model, ignore_layer=None, scaling_factor=None, act_num_bits=None, weight_num_bits=None):
        """
        Entry point of quantizing the model: this function is used to filter out the ignored layers and initialize the replace map.
        :param model: model returned from "prepare_model" function
        :param ignore_layer: layers that we don't want to be quantized
        :return: no return value
        """
        if scaling_factor is None:
            raise ValueError("Please specify the scaling_factor parameter!")
        if act_num_bits is None or weight_num_bits is None:
            raise ValueError("Please specify the num_bits parameter!")
        module_dict = {}
        for entry in quant_modules._DEFAULT_QUANT_MAP:
            module = getattr(entry.orig_mod, entry.mod_name)
            # if entry.mod_name == 'Conv2d':
            #     module_dict[id(module)] = MyQuantConv2d
            # else: 
            module_dict[id(module)] = entry.replace_mod
        test_torch_module_find_quant_module(model, module_dict, ignore_layer, current_path=None, scaling_factor=scaling_factor, act_num_bits=act_num_bits, weight_num_bits=weight_num_bits)

    def transfer_torch_to_quantization(nn_instance, quant_mudule):
        """
        This function mainly instanciate the quantized layer,
        migrate all the attributes of the original layer to the quantized layer,
        and add the default descriptor to the quantized layer, i.e., how should weight and quantization be quantized.
        :param nn_instance: original layer
        :param quant_mudule: corresponding class of the quantized layer
        :return: no return value
        """
        quant_instance = quant_mudule.__new__(quant_mudule)
        for k, val in vars(nn_instance).items():
            setattr(quant_instance, k, val)

        def __init__(self):
            # Return two instances of QuantDescriptor; self.__class__ is the class of quant_instance, E.g.: QuantConv2d
            quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
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
    
    def transfer_test_torch_to_quantization(nn_instance, quant_mudule, act_num_bits=None, weight_num_bits=None):
        """
        This function mainly instanciate the quantized layer,
        migrate all the attributes of the original layer to the quantized layer,
        and add the default descriptor to the quantized layer, i.e., how should weight and quantization be quantized.
        :param nn_instance: original layer
        :param quant_mudule: corresponding class of the quantized layer
        :return: no return value
        """
        quant_instance = quant_mudule.__new__(quant_mudule)
        for k, val in vars(nn_instance).items():
            setattr(quant_instance, k, val)

        def __init__(self):
            # Return two instances of QuantDescriptor; self.__class__ is the class of quant_instance, E.g.: QuantConv2d
            # quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
            quant_desc_input = QuantDescriptor(
                num_bits=act_num_bits,
                # unsigned=True
            )
            quant_desc_weight = QuantDescriptor(
                num_bits=weight_num_bits, 
                axis=(0), 
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
    
    def my_transfer_torch_to_quantization(nn_instance, quant_mudule, scaling_factor=None, act_num_bits=None, weight_num_bits=None):
        """
        This function mainly instanciate the quantized layer,
        migrate all the attributes of the original layer to the quantized layer,
        and add the default descriptor to the quantized layer, i.e., how should weight and quantization be quantized.
        :param nn_instance: original layer
        :param quant_mudule: corresponding class of the quantized layer
        :return: no return value
        """
        if scaling_factor is None:
            raise ValueError("Please specify the scaling_factor parameter!")
        if act_num_bits is None or weight_num_bits is None:
            raise ValueError("Please specify the num_bits parameter!")
        quant_instance = quant_mudule.__new__(quant_mudule)
        for k, val in vars(nn_instance).items():
            if isinstance(val, tuple):
                val = val[0]
            setattr(quant_instance, k, val)
        my_activation_descriptor = QuantDescriptor(
            num_bits=act_num_bits,
            # unsigned=True
        )
        my_weight_descriptor = QuantDescriptor(
            num_bits=weight_num_bits, 
            axis=(0), 
        )
        my_activation_quantizer = TensorQuantizer(my_activation_descriptor)
        my_weight_quantizer = TensorQuantizer(my_weight_descriptor)
        quant_instance._input_quantizer = my_activation_quantizer
        quant_instance._weight_quantizer = my_weight_quantizer
        quant_instance.scaling_factor = scaling_factor
        return quant_instance
    
    def vali_original_head_torch_module_find_quant_module(module, module_dict, ignore_layer, current_path='', scaling_factor=None, act_num_bits=None, weight_num_bits=None):
        """
        This is a recursive function that finds all modules within the model
        and replaces the modules with their quantized versions by calling
        "transfer_torch_to_quantization" function if conditions are met.
        :param module: The model to be quantized
        :param module_dict: Replace map, with key being module layer id and value being the class of the quantized layer
        :param ignore_layer: List of layers to be ignored
        :param current_path: Accumulated path of the module names for nested structure tracking
        :return: no return value
        """
        if scaling_factor is None:
            raise ValueError("Please specify the scaling_factor parameter!")

        if act_num_bits is None or weight_num_bits is None:
            raise ValueError("Please specify the num_bits parameter!")

        if module is None:
            return

        for name, submodule in module.named_children():
            # Update the path for each submodule
            path = f"{current_path}.{name}" if current_path else name
            # Recursively process each submodule
            vali_original_head_torch_module_find_quant_module(submodule, module_dict, ignore_layer, path, scaling_factor, act_num_bits, weight_num_bits)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict and not quantization_ignore_match(ignore_layer, path):
                no_list = ['dense_head.heads_list.0.center.1',
                        'dense_head.heads_list.0.center_z.1',
                        'dense_head.heads_list.0.dim.1',
                        'dense_head.heads_list.0.rot.1',
                        'dense_head.heads_list.0.vel.1',
                        'dense_head.heads_list.0.hm.0.0',
                        'dense_head.heads_list.0.hm.1',
                        'dense_head.heads_list.1.center.1',
                        'dense_head.heads_list.1.center_z.1',
                        'dense_head.heads_list.1.dim.1',
                        'dense_head.heads_list.1.rot.1',
                        'dense_head.heads_list.1.vel.1',
                        'dense_head.heads_list.1.hm.0.0',
                        'dense_head.heads_list.1.hm.1',
                        'dense_head.heads_list.2.center.1',
                        'dense_head.heads_list.2.dim.1',
                        'dense_head.heads_list.2.rot.1',
                        'dense_head.heads_list.2.vel.1',
                        'dense_head.heads_list.2.hm.0.0',
                        'dense_head.heads_list.2.hm.1',
                        'dense_head.heads_list.3.center.1',
                        'dense_head.heads_list.3.dim.1',
                        'dense_head.heads_list.3.vel.1',
                        'dense_head.heads_list.3.hm.0.0',
                        'dense_head.heads_list.3.hm.1',
                        'dense_head.heads_list.4.center.1',
                        'dense_head.heads_list.4.dim.1',
                        'dense_head.heads_list.4.vel.1',
                        'dense_head.heads_list.4.hm.0.0',
                        'dense_head.heads_list.4.hm.1',
                        'dense_head.heads_list.5.center.1',
                        'dense_head.heads_list.5.dim.1',
                        'dense_head.heads_list.5.vel.1',
                        'dense_head.heads_list.5.hm.0.0',
                        'dense_head.heads_list.5.hm.1']
                # Check if the module is within the specified path and is a Conv2d for special handling
                if 'dense_head' in path and isinstance(submodule, torch.nn.Conv2d) and path not in no_list or 'backbone_2d' in path and isinstance(submodule, torch.nn.Conv2d):
                    # module._modules[name] = transfer_test_torch_to_quantization(submodule, quant_nn.Conv2d, act_num_bits, weight_num_bits)
                    module._modules[name] = my_transfer_torch_to_quantization(submodule, MyQuantConv2d, scaling_factor, act_num_bits, weight_num_bits)
                # else:
                #     # General case: replace the submodule with its quantized version
                #     module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])
    
    def test_head_torch_module_find_quant_module(module, module_dict, ignore_layer, current_path='', scaling_factor=None, act_num_bits=None, weight_num_bits=None):
        """
        This is a recursive function that finds all modules within the model
        and replaces the modules with their quantized versions by calling
        "transfer_torch_to_quantization" function if conditions are met.
        :param module: The model to be quantized
        :param module_dict: Replace map, with key being module layer id and value being the class of the quantized layer
        :param ignore_layer: List of layers to be ignored
        :param current_path: Accumulated path of the module names for nested structure tracking
        :return: no return value
        """
        if scaling_factor is None:
            raise ValueError("Please specify the scaling_factor parameter!")

        if act_num_bits is None or weight_num_bits is None:
            raise ValueError("Please specify the num_bits parameter!")

        if module is None:
            return

        for name, submodule in module.named_children():
            # Update the path for each submodule
            path = f"{current_path}.{name}" if current_path else name
            # Recursively process each submodule
            test_head_torch_module_find_quant_module(submodule, module_dict, ignore_layer, path, scaling_factor, act_num_bits, weight_num_bits)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict and not quantization_ignore_match(ignore_layer, path):
                no_list = ['dense_head.heads_list.0.center.1',
                        'dense_head.heads_list.0.center_z.1',
                        'dense_head.heads_list.0.dim.1',
                        'dense_head.heads_list.0.rot.1',
                        'dense_head.heads_list.0.vel.1',
                        'dense_head.heads_list.0.hm.0.0',
                        'dense_head.heads_list.0.hm.1',
                        'dense_head.heads_list.1.center.1',
                        'dense_head.heads_list.1.center_z.1',
                        'dense_head.heads_list.1.dim.1',
                        'dense_head.heads_list.1.rot.1',
                        'dense_head.heads_list.1.vel.1',
                        'dense_head.heads_list.1.hm.0.0',
                        'dense_head.heads_list.1.hm.1',
                        'dense_head.heads_list.2.center.1',
                        'dense_head.heads_list.2.dim.1',
                        'dense_head.heads_list.2.rot.1',
                        'dense_head.heads_list.2.vel.1',
                        'dense_head.heads_list.2.hm.0.0',
                        'dense_head.heads_list.2.hm.1',
                        'dense_head.heads_list.3.center.1',
                        'dense_head.heads_list.3.dim.1',
                        'dense_head.heads_list.3.vel.1',
                        'dense_head.heads_list.3.hm.0.0',
                        'dense_head.heads_list.3.hm.1',
                        'dense_head.heads_list.4.center.1',
                        'dense_head.heads_list.4.dim.1',
                        'dense_head.heads_list.4.vel.1',
                        'dense_head.heads_list.4.hm.0.0',
                        'dense_head.heads_list.4.hm.1',
                        'dense_head.heads_list.5.center.1',
                        'dense_head.heads_list.5.dim.1',
                        'dense_head.heads_list.5.vel.1',
                        'dense_head.heads_list.5.hm.0.0',
                        'dense_head.heads_list.5.hm.1']
                # Check if the module is within the specified path and is a Conv2d for special handling
                if 'dense_head' in path and isinstance(submodule, torch.nn.Conv2d) and path not in no_list or 'backbone_2d' in path and isinstance(submodule, torch.nn.Conv2d):
                    module._modules[name] = my_transfer_torch_to_quantization(submodule, MyQuantConv2d, scaling_factor, act_num_bits, weight_num_bits)
                else:
                    # General case: replace the submodule with its quantized version
                    module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    def test_torch_module_find_quant_module(module, module_dict, ignore_layer, current_path='', scaling_factor=None, act_num_bits=None, weight_num_bits=None):
        """
        This is a recursive function that finds all modules within the model
        and replaces the modules with their quantized versions by calling
        "transfer_torch_to_quantization" function if conditions are met.
        :param module: The model to be quantized
        :param module_dict: Replace map, with key being module layer id and value being the class of the quantized layer
        :param ignore_layer: List of layers to be ignored
        :param current_path: Accumulated path of the module names for nested structure tracking
        :return: no return value
        """
        if scaling_factor is None:
            raise ValueError("Please specify the scaling_factor parameter!")

        if act_num_bits is None or weight_num_bits is None:
            raise ValueError("Please specify the num_bits parameter!")

        if module is None:
            return

        for name, submodule in module.named_children():
            # Update the path for each submodule
            path = f"{current_path}.{name}" if current_path else name

            # Recursively process each submodule
            test_torch_module_find_quant_module(submodule, module_dict, ignore_layer, path, scaling_factor, act_num_bits, weight_num_bits)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict and not quantization_ignore_match(ignore_layer, path):
                # Check if the module is within the specified path and is a Conv2d for special handling
                if 'backbone_2d' in path and isinstance(submodule, torch.nn.Conv2d):
                    module._modules[name] = my_transfer_torch_to_quantization(submodule, MyQuantConv2d, scaling_factor, act_num_bits, weight_num_bits)
                else:
                    # General case: replace the submodule with its quantized version
                    module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    def test_original_torch_module_find_quant_module(module, module_dict, ignore_layer, current_path='', scaling_factor=None, act_num_bits=None, weight_num_bits=None):
        """
        This is a recursive function that finds all modules within the model
        and replaces the modules with their quantized versions by calling
        "transfer_torch_to_quantization" function if conditions are met.
        :param module: The model to be quantized
        :param module_dict: Replace map, with key being module layer id and value being the class of the quantized layer
        :param ignore_layer: List of layers to be ignored
        :param current_path: Accumulated path of the module names for nested structure tracking
        :return: no return value
        """
        if scaling_factor is None:
            raise ValueError("Please specify the scaling_factor parameter!")

        if act_num_bits is None or weight_num_bits is None:
            raise ValueError("Please specify the num_bits parameter!")

        if module is None:
            return

        for name, submodule in module.named_children():
            # Update the path for each submodule
            path = f"{current_path}.{name}" if current_path else name

            # Recursively process each submodule
            test_original_torch_module_find_quant_module(submodule, module_dict, ignore_layer, path, scaling_factor, act_num_bits, weight_num_bits)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict and not quantization_ignore_match(ignore_layer, path):
                # # Check if the module is within the specified path and is a Conv2d for special handling
                # if 'dense_head' in path and isinstance(submodule, torch.nn.Conv2d):
                #     module._modules[name] = transfer_test_torch_to_quantization(submodule, quant_nn.Conv2d, act_num_bits, weight_num_bits)
                # else:
                no_list = ['dense_head.heads_list.0.center.1',
                        'dense_head.heads_list.0.center_z.1',
                        'dense_head.heads_list.0.dim.1',
                        'dense_head.heads_list.0.rot.1',
                        'dense_head.heads_list.0.vel.1',
                        'dense_head.heads_list.0.hm.0.0',
                        'dense_head.heads_list.0.hm.1',
                        'dense_head.heads_list.1.center.1',
                        'dense_head.heads_list.1.center_z.1',
                        'dense_head.heads_list.1.dim.1',
                        'dense_head.heads_list.1.rot.1',
                        'dense_head.heads_list.1.vel.1',
                        'dense_head.heads_list.1.hm.0.0',
                        'dense_head.heads_list.1.hm.1',
                        'dense_head.heads_list.2.center.1',
                        'dense_head.heads_list.2.dim.1',
                        'dense_head.heads_list.2.rot.1',
                        'dense_head.heads_list.2.vel.1',
                        'dense_head.heads_list.2.hm.0.0',
                        'dense_head.heads_list.2.hm.1',
                        'dense_head.heads_list.3.center.1',
                        'dense_head.heads_list.3.dim.1',
                        'dense_head.heads_list.3.vel.1',
                        'dense_head.heads_list.3.hm.0.0',
                        'dense_head.heads_list.3.hm.1',
                        'dense_head.heads_list.4.center.1',
                        'dense_head.heads_list.4.dim.1',
                        'dense_head.heads_list.4.vel.1',
                        'dense_head.heads_list.4.hm.0.0',
                        'dense_head.heads_list.4.hm.1',
                        'dense_head.heads_list.5.center.1',
                        'dense_head.heads_list.5.dim.1',
                        'dense_head.heads_list.5.vel.1',
                        'dense_head.heads_list.5.hm.0.0',
                        'dense_head.heads_list.5.hm.1']
                # Check if the module is within the specified path and is a Conv2d for special handling
                if 'dense_head' in path and isinstance(submodule, torch.nn.Conv2d) and path not in no_list or 'backbone_2d' in path and isinstance(submodule, torch.nn.Conv2d):
                    module._modules[name] = transfer_test_torch_to_quantization(submodule, quant_nn.Conv2d, act_num_bits, weight_num_bits)
                    # module._modules[name] = my_transfer_torch_to_quantization(submodule, MyQuantConv2d, scaling_factor, act_num_bits, weight_num_bits)
                    # module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])
                    # General case: replace the submodule with its quantized version
                    # module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    def torch_module_find_quant_module(module, module_dict, ignore_layer):
        """
        This is a recursive function that find all modules within the map and also within the model
        and replace the module to the quantized version by call "transfer_torch_to_quantization" function
        :param module: The model to be quantized
        :param module_dict: Replace map, with key is module layer id and value is the class of the quantized layer
        :param ignore_layer: List of layers to be ignored
        :return: no return value
        """
        if module is None:
            return
        
        for name in module._modules:
            submodule = module._modules[name]
            path = name
            torch_module_find_quant_module(submodule, module_dict, ignore_layer)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                ignored = quantization_ignore_match(ignore_layer, path)
                if ignored:
                    print(f"Quantization : {path} has ignored.")
                    continue
                # substitute the layer with quantized version
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    def quantization_ignore_match(ignore_layer, path):
        """
        A simple function to determine whether a layer is ignored
        """
        if ignore_layer is None:
            return False
        if isinstance(ignore_layer, str) or isinstance(ignore_layer, list):
            if isinstance(ignore_layer, str):
                ignore_layer = [ignore_layer]
            if path in ignore_layer:
                return True
            for item in ignore_layer:
                if re.match(item, path):
                    return True
        return False
    
    def compute_amax(model, device, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(**kwargs)
                    module._amax = module._amax.to(device)

    def compute_amax_weight_only(model, device, **kwargs):
        for name, module in model.named_modules():
            if name.endswith('_weight_quantizer'):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)
                    module._amax = module._amax.to(device)


    # method parameter designed for "histogram", method in ['entropy', 'mse', 'percentile']
    def calibrate_model(model, dataloader, device, method):
        # Collect stats with data flowing
        collect_stats(model, dataloader, device)
        # Get dynamic range and compute amax (used in calibration)
        compute_amax(model, device, method=method)

    def quantize_model(model, ignore_layer=None, calib_method="max"):
        """
        This function is used to quantize the model
        :param model: model returned from "prepare_model" function
        :param ignore_layer: layers that we don't want to be quantized
        :return: no return value
        """
        initialize(calib_method)
        replace_to_quantization_model(model, ignore_layer)

    def quantize_model_unsigned(model, ignore_layer=None, calib_method="max"):
        """
        This function is used to quantize the model
        :param model: model returned from "prepare_model" function
        :param ignore_layer: layers that we don't want to be quantized
        :return: no return value
        """
        initialize_unsigned_for_input(calib_method)
        replace_to_quantization_model(model, ignore_layer)

    def quantize_model_unsigned_convtranspose(model, ignore_layer=None, calib_method="max"):
        """
        This function is used to quantize the model
        :param model: model returned from "prepare_model" function
        :param ignore_layer: layers that we don't want to be quantized
        :return: no return value
        """
        initialize_unsigned_for_input_and_convtranspose(calib_method)
        replace_to_quantization_model(model, ignore_layer)

    def my_quantize_model(model, ignore_layer=None, calib_method="max", scaling_factor=0.5, act_num_bits=None, weight_num_bits=None):
        """
        This function is used to quantize the model
        :param model: model returned from "prepare_model" function
        :param ignore_layer: layers that we don't want to be quantized
        :return: no return value
        """
        initialize(calib_method)
        my_replace_to_quantization_model(model, ignore_layer, scaling_factor, act_num_bits, weight_num_bits)

    def my_original_quantize_model(model, ignore_layer=None, calib_method="max", scaling_factor=0.5, act_num_bits=None, weight_num_bits=None):
        """
        This function is used to quantize the model
        :param model: model returned from "prepare_model" function
        :param ignore_layer: layers that we don't want to be quantized
        :return: no return value
        """
        initialize_all(calib_method, num_bits=16)
        my_original_replace_to_quantization_model(model, ignore_layer, scaling_factor, act_num_bits, weight_num_bits)

    def my_head_quantize_model(model, ignore_layer=None, calib_method="max", scaling_factor=0.5, act_num_bits=None, weight_num_bits=None):
        """
        This function is used to quantize the model
        :param model: model returned from "prepare_model" function
        :param ignore_layer: layers that we don't want to be quantized
        :return: no return value
        """
        initialize_all(calib_method, num_bits=8)
        my_head_replace_to_quantization_model(model, ignore_layer, scaling_factor, act_num_bits, weight_num_bits)

    def vali_quantize_model(model, ignore_layer=None, calib_method="max", scaling_factor=0.5, act_num_bits=None, weight_num_bits=None):
        """
        This function is used to quantize the model
        :param model: model returned from "prepare_model" function
        :param ignore_layer: layers that we don't want to be quantized
        :return: no return value
        """
        initialize_all(calib_method, num_bits=16)
        valid_replace_to_quantization_model(model, ignore_layer, scaling_factor, act_num_bits, weight_num_bits)

    def collect_stats(model, data_loader, num_batch=200):
        model.eval()
        for name, module in model.named_modules():
            if name.endswith('_quantizer'):
                module.enable_calib()
                module.disable_quant()

        with torch.no_grad():
            for i, batch_dict in enumerate(tqdm(data_loader, desc='calibration', total=num_batch+1)):
                load_data_to_gpu(batch_dict)
                model(batch_dict)
                if i > num_batch:
                    break

        for name, module in model.named_modules():
            if name.endswith('_quantizer'):
                module.disable_calib()
                module.enable_quant()

    def collect_stats_weight_only(model, data_loader, num_batch=200):
        model.eval()
        for name, module in model.named_modules():
            if name.endswith('_weight_quantizer'):
                module.enable_calib()
                module.disable_quant()

        with torch.no_grad():
            for i, batch_dict in enumerate(tqdm(data_loader, desc='calibration', total=num_batch+1)):
                load_data_to_gpu(batch_dict)
                model(batch_dict)
                if i > num_batch:
                    break

        for name, module in model.named_modules():
            if name.endswith('_weight_quantizer'):
                module.disable_calib()
                module.enable_quant()


    def get_accuracy_graph(ignore_layer=None, quant=None):

        if quant is None:
            raise ValueError("Please specify the quant parameter!")

        num_bits_list = [16, 8, 4, 3, 2]

        for num_bits in num_bits_list:

            logger.info('Building model...')

            test_set, test_loader, sampler = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                batch_size=args.batch_size,
                dist=dist_test, workers=args.workers, logger=logger, training=False
            )

            model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

            model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

            model.cuda()

            logger.info('Model built successfully!')

            logger.info(f"Quantizing model's {quant} with {num_bits} bits...")
            if quant == 'weight':
                initialize_weight("max", num_bits)
            elif quant == 'input':
                initialize_input("max", num_bits)
            else:
                raise ValueError("Invalid quant parameter!")
            logger.info('Replacing module...')
            replace_to_quantization_model(model, ignore_layer)
            logger.info('Collecting stats...')
            collect_stats(model, test_loader, 200)
            logger.info('Computing amax...')
            compute_amax(model, device, method='entropy')

            model.cuda()
            logger.info('Model quantized successfully!')
            logger.info('Evaluating model...')
            eval_utils.eval_one_epoch(
                cfg, args, model, test_loader, epoch_id, logger, dist_test=False,
                result_dir=eval_output_dir
            )
            logger.info(model)


    # get_accuracy_graph(ignore_layer=None, quant='weight')
    # get_accuracy_graph(ignore_layer=None, quant='input')
    
    def register_collect_smoothquant_hook(model, data_loader, num_batch=200):
        model.eval()
        act_scales = {}
        weight_scales = {}

        def forward_hook(module, input, name):
            hidden_dim_act = input[0].shape[1]
            tensor_act = input[0].view(-1, hidden_dim_act).abs().detach()
            comming_max_act = torch.max(tensor_act, dim=0)[0].float().cpu()
            if name not in act_scales:
                act_scales[name] = comming_max_act
            else:
                act_scales[name] = torch.max(act_scales[name], comming_max_act)

            hidden_dim_weight = module.weight.shape[1]
            tensor_weight = module.weight.view(-1, hidden_dim_weight).abs().detach()
            comming_max_weight = torch.max(tensor_weight, dim=0)[0].float().cpu()
            if name not in weight_scales:
                weight_scales[name] = comming_max_weight
            else:
                weight_scales[name] = torch.max(weight_scales[name], comming_max_weight)


        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hook = module.register_forward_pre_hook(partial(forward_hook, name=name))
                hooks.append(hook)

        try:
            with torch.no_grad():
                for i, inputs in enumerate(tqdm(data_loader, desc='collecting stats', total=num_batch)):
                    if i >= num_batch:
                        break
                    load_data_to_gpu(inputs)
                    model(inputs)
        finally:
            for h in hooks:
                h.remove()
        return act_scales, weight_scales
    
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    # test_set, test_loader, sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=args.batch_size,
    #     dist=dist_test, workers=args.workers, logger=logger, training=False
    # )

    # model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    # model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

    # model.cuda()

    # act_scales, weight_scales = register_collect_smoothquant_hook(model, test_loader, 200)

    # scales = {}

    # for name, act_scale, weight_scale in zip(act_scales.keys(), act_scales.values(), weight_scales.values()):
    #     scale = torch.sqrt(act_scale / weight_scale)
    #     scales[name] = scale.view(1, -1, 1, 1).to(device)

    def register_smoothquant_act_hook(model, scales):
        def forward_pre_hook(module, input, name):
            modified_input = input[0] / scales[name]

            return (modified_input,)

        handles = []
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.Conv2d):
                handle = module.register_forward_pre_hook(partial(forward_pre_hook, name=name))
                handles.append(handle)
        return handles
    
    def register_smoothquant_weight_hook(model, scales):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.Conv2d):
                with torch.no_grad():
                    module.weight *= scales[name]

        return

    def register_collect_my_input_hook(model):

        def forward_hook(module, input, output, name):
            global my_res
            my_res[name]=output.detach()

        for name, module in model.named_modules():
            if isinstance(module, MyQuantConv2d) or isinstance(module, torch.nn.Conv2d):
                module.register_forward_hook(partial(forward_hook, name=name))

    def register_collect_input_hook(model):

        def forward_hook(module, input, output, name):
            global res
            res[name]=output.detach()
            
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module.register_forward_hook(partial(forward_hook, name=name))

    def register_collect_quant_input_hook(model):

        def forward_hook(module, input, output, name):
            global quant_res
            quant_res[name]=output.detach()
            
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.Conv2d) or isinstance(module, torch.nn.Conv2d):
                module.register_forward_hook(partial(forward_hook, name=name))
                

    # quantize_model(model, ignore_layer=None, calib_method="max")

    # hooks = register_smoothquant_act_hook(model, scales)

    # register_smoothquant_weight_hook(model, scales)

    # collect_stats(model, test_loader, 200)

    # compute_amax(model, device, method='entropy')

    # model.cuda()

    # eval_utils.eval_one_epoch(
    #     cfg, args, model, test_loader, epoch_id, logger, dist_test=False,
    #     result_dir=eval_output_dir
    # )

    # for h in hooks:
    #     h.remove()

    # logger.info(model)

    # logger.info('--------Above is the SmoothQuanted model results--------')

    def get_l1_loss():
        my_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

        my_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

        my_model.cuda()

        vali_quantize_model(my_model, ignore_layer=None, calib_method="max", scaling_factor=0.5, act_num_bits=8, weight_num_bits=8)

        register_collect_my_input_hook(my_model)

        with torch.no_grad():
            for i, batch_dict in enumerate(tqdm(test_loader, desc='calibration', total=1)):
                load_data_to_gpu(batch_dict)
                my_model(batch_dict)
                break

        # logger.info(my_model)

        # logger.info('--------Above is the SmoothQuanted model results--------')

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

        model.cuda()

        register_collect_input_hook(model)

        with torch.no_grad():
            for i, batch_dict in enumerate(tqdm(test_loader, desc='calibration', total=1)):
                load_data_to_gpu(batch_dict)
                model(batch_dict)
                break

        for key in my_res.keys():
            logger.info(f'name: {key}, l1loss: {torch.nn.L1Loss()(my_res[key], res[key]).item()}') 

        logger.info('--------Above is the l1loss between final model and the original one--------')

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

        model.cuda()

        quantize_model(model, ignore_layer=None, calib_method="max")

        register_collect_quant_input_hook(model)

        # logger.info(model)

        with torch.no_grad():
            for i, batch_dict in enumerate(tqdm(test_loader, desc='calibration', total=1)):
                load_data_to_gpu(batch_dict)
                model(batch_dict)
                break

        model.cuda()

        for key in my_res.keys():
            logger.info(f'name: {key}, l1loss: {torch.nn.L1Loss()(quant_res[key], res[key]).item()}') 

        exit()

    # get_l1_loss()

    def evaluate_with_scale():
        scale_list = torch.arange(0.1, 1, 0.05)
        for scale in scale_list:
            my_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

            my_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

            my_model.cuda()

            my_quantize_model(my_model, ignore_layer=None, calib_method="max", scaling_factor=scale)

            collect_stats(my_model, test_loader, 200)

            compute_amax(my_model, device, method='entropy')

            my_model.cuda()

            eval_utils.eval_one_epoch(
                cfg, args, my_model, test_loader, epoch_id, logger, dist_test=False,
                result_dir=eval_output_dir
            )

            logger.info(my_model)

            logger.info(f'********Above is the results if scaling factor = {scale.item()}********')

    # evaluate_with_scale()

    def evaluate_with_bits():
        num_bits_list = [16, 8, 4, 3, 2]
        for act_num_bits in num_bits_list:
            for weight_num_bits in num_bits_list:
                my_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

                my_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

                my_model.cuda()



                my_quantize_model(my_model, ignore_layer=None, calib_method="max", act_num_bits=act_num_bits, weight_num_bits=weight_num_bits)

                collect_stats(my_model, test_loader, 200)

                compute_amax(my_model, device, method='entropy')

                my_model.cuda()

                eval_utils.eval_one_epoch(
                    cfg, args, my_model, test_loader, epoch_id, logger, dist_test=False,
                    result_dir=eval_output_dir
                )

                logger.info(my_model)

                logger.info(f'********Above is the results if activation bits = {act_num_bits} and weight bits = {weight_num_bits}********')

    def evaluate_with_original_bits():
        
        my_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

        my_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

        my_model.cuda()

        my_original_quantize_model(my_model, ignore_layer=None, calib_method="max", scaling_factor=0.5, act_num_bits=8, weight_num_bits=8)

        collect_stats(my_model, test_loader, 200)

        compute_amax(my_model, device, method='entropy')

        my_model.cuda()

        eval_utils.eval_one_epoch(
            cfg, args, my_model, test_loader, epoch_id, logger, dist_test=False,
            result_dir=eval_output_dir
        )

        logger.info(my_model)

    evaluate_with_original_bits()

    def evaluate_final_model():
        my_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

        my_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

        my_model.cuda()

        my_head_quantize_model(my_model, ignore_layer=None, calib_method="max", scaling_factor=0.5, act_num_bits=8, weight_num_bits=8)

        collect_stats(my_model, test_loader, 200)

        compute_amax(my_model, device, method='entropy')

        my_model.cuda()

        eval_utils.eval_one_epoch(
            cfg, args, my_model, test_loader, epoch_id, logger, dist_test=False,
            result_dir=eval_output_dir
        )

        logger.info(my_model)

    def my_original_evaluate_with_bits():
        num_bits_list = [16, 8, 4, 3, 2]
        for act_num_bits in num_bits_list:
            for weight_num_bits in num_bits_list:
                my_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

                my_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

                my_model.cuda()



                quantize_model(my_model, ignore_layer=None, calib_method="max", act_num_bits=act_num_bits, weight_num_bits=weight_num_bits)

                collect_stats(my_model, test_loader, 200)

                compute_amax(my_model, device, method='entropy')

                my_model.cuda()

                eval_utils.eval_one_epoch(
                    cfg, args, my_model, test_loader, epoch_id, logger, dist_test=False,
                    result_dir=eval_output_dir
                )

                logger.info(my_model)

                logger.info(f'********Above is the results if activation bits = {act_num_bits} and weight bits = {weight_num_bits}********')

    def my_evaluate_with_bits():
        num_bits_list = [16, 8, 4, 3, 2]
        for act_num_bits in num_bits_list:
            for weight_num_bits in num_bits_list:

                my_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

                my_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

                my_model.cuda()



                my_head_quantize_model(my_model, ignore_layer=None, calib_method="max", act_num_bits=act_num_bits, weight_num_bits=weight_num_bits)

                collect_stats(my_model, test_loader, 200)

                compute_amax(my_model, device, method='entropy')

                my_model.cuda()

                eval_utils.eval_one_epoch(
                    cfg, args, my_model, test_loader, epoch_id, logger, dist_test=False,
                    result_dir=eval_output_dir
                )

                logger.info(my_model)

                logger.info(f'********Above is the results if activation bits = {act_num_bits} and weight bits = {weight_num_bits}********')

    def validate_model():

        my_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

        my_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False, pre_trained_path=args.pretrained_model)

        my_model.cuda()

        vali_quantize_model(my_model, ignore_layer=None, calib_method="max", scaling_factor=0.5, act_num_bits=8, weight_num_bits=8)

        collect_stats(my_model, test_loader, 200)

        compute_amax(my_model, device, method='entropy')

        my_model.cuda()

        eval_utils.eval_one_epoch(
            cfg, args, my_model, test_loader, epoch_id, logger, dist_test=False,
            result_dir=eval_output_dir
        )

        logger.info(my_model)

    validate_model()

    exit()

    # quantize_model(model, ignore_layer=None, calib_method="max")

    # collect_stats(model, test_loader, 200)

    # compute_amax(model, device, method='entropy')

    # model.cuda()

    # eval_utils.eval_one_epoch(
    #     cfg, args, model, test_loader, epoch_id, logger, dist_test=False,
    #     result_dir=eval_output_dir
    # )

    # logger.info(model)

    exit()

    print(prof.key_averages().table(sort_by="gpu_memory_usage", row_limit=10))

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()
