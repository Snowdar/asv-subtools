# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-05-29)

import sys, os
import math
import yaml
import random
import logging
import shutil
import copy
import numpy as np
import pandas as pd
import platform
import csv
import time
import torch
import torch.distributed as dist

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def to_bool(variable):
    """Transform string to bool if variable is not bool
    """
    if not isinstance(variable, bool):
        if not isinstance(variable, str):
            raise TypeError("variable is not str or bool type")
        else:
            return True if variable == 'true' or variable == 'True' else False
    else:
        return variable


def parse_gpu_id_option(gpu_id):
    """
    @gpu_id: str: 1,2,3 or 1-2-3 or "1 2 3"
              int: 1
              list/tuple: [1,2,3] or ("1","2","3")
    """
    if isinstance(gpu_id, str):
        gpu_id = gpu_id.replace("-", " ")
        gpu_id = gpu_id.replace(",", " ")
        gpu_id = [ int(x) for x in gpu_id.split()]
    elif isinstance(gpu_id, int):
        gpu_id = [gpu_id]
    elif isinstance(gpu_id, (list, tuple)):
        gpu_id = [ int(x) for x in gpu_id ]
    else:
        raise TypeError("Expected str, int or list/tuple, bug got {}.".format(gpu_id))
    return gpu_id


def select_model_device(model, use_gpu, gpu_id="", benchmark=False):
    """ Auto select device (cpu/GPU) for model
    @use_gpu: bool or 'true'/'false' string
    """
    model.cpu()
    
    use_gpu = to_bool(use_gpu)
    benchmark = to_bool(benchmark)

    if use_gpu :
        torch.backends.cudnn.benchmark = benchmark

        if gpu_id == "":
            logger.info("The use_gpu is true and gpu id is not specified, so select gpu device automatically.")
            import libs.support.GPU_Manager as gpu
            gm = gpu.GPUManager()
            gpu_id = [gm.auto_choice()]
        else:
            # Get a gpu id list.
            gpu_id = parse_gpu_id_option(gpu_id)
            if is_main_training(): logger.info("The use_gpu is true and training will use GPU {0}.".format(gpu_id))

        ## Multi-GPU with DDP.
        if len(gpu_id) > 0 and use_ddp():
            if dist.get_world_size() != len(gpu_id):
                raise ValueError("To run DDP with {} nj, " \
                                 "but {} GPU ids ({}) are given.".format(dist.get_world_size(), len(gpu_id), gpu_id))
            torch.cuda.set_device(gpu_id[dist.get_rank()])
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id[dist.get_rank()]], output_device=dist.get_rank())
            return model

        ## Multi-GPU with Horovod.
        if len(gpu_id) > 1 and use_horovod():
            import horovod.torch as hvd
            # Just multi GPU case.
            if hvd.size() != len(gpu_id):
                raise ValueError("To run horovod with {} nj, " \
                                 "but {} GPU ids ({}) are given.".format(hvd.size(), len(gpu_id), gpu_id))
            torch.cuda.set_device(gpu_id[hvd.rank()])
        else:
            ## One process in one GPU.
            torch.cuda.set_device(gpu_id[0])

        model.cuda()

    return model


def to_device(device_object, tensor):
    """
    Select device for non-parameters tensor w.r.t model or tensor which has been specified a device.
    """
    if isinstance(device_object, torch.nn.Module):
        device = next(device_object.parameters()).device
    elif isinstance(device_object, torch.Tensor):
        device = device_object.device

    return tensor.to(device)


def get_device(model):
    assert isinstance(model, torch.nn.Module)
    device = next(model.parameters()).device
    return device


def get_device_from_optimizer(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            return p.device


def get_tensors(tensor_sets):
    """Get a single tensor list from a nested tensor_sets list/tuple object,
    such as transforming [(tensor1,tensor2),tensor3] to [tensor1,tensor2,tensor3]
    """
    tensors = []
    
    for this_object in tensor_sets:
        # Only tensor
        if isinstance(this_object, torch.Tensor):
            tensors.append(this_object)
        if isinstance(this_object, np.ndarray):
            tensors.append(torch.from_numpy(this_object))
        elif isinstance(this_object, list) or isinstance(this_object, tuple):
            tensors.extend(get_tensors(this_object))

    return tensors


def for_device_free(function):
    """
    A decorator to make class-function with input-tensor device-free
    Used in libs.nnet.framework.TopVirtualNnet
    """
    def wrapper(self, *tensor_sets):
        transformed = []

        for tensor in get_tensors(tensor_sets):
            transformed.append(to_device(self, tensor))

        return function(self, *transformed)

    return wrapper


def create_model_from_py(model_blueprint, model_creation=""):
    """ Used in pipeline/train.py and pipeline/onestep/extract_emdeddings.py and it makes config of nnet
    more free with no-change of training and other common scripts.

    @model_blueprint: string type, a *.py file path which includes the instance of nnet, such as examples/xvector.py
    @model_creation: string type, a command to create the model class according to the class declaration 
                     in model_blueprint, such as using 'Xvector(40,2)' to create an Xvector nnet.
                     Note, it will return model_module if model_creation is not given, else return model.
    """
    if not os.path.exists(model_blueprint):
        raise TypeError("Expected {} to exist.".format(model_blueprint))
    if os.path.getsize(model_blueprint) == 0:
        raise TypeError("There is nothing in {}.".format(model_blueprint))

    sys.path.insert(0, os.path.dirname(model_blueprint))
    model_module_name = os.path.basename(model_blueprint).split('.')[0]
    model_module = __import__(model_module_name)

    if model_creation == "":
        return model_module
    else:
        model = eval("model_module.{0}".format(model_creation))

        return model


def write_nnet_config(model_blueprint:str, model_creation:str, nnet_config:str):
    dataframe = pd.DataFrame([model_blueprint, model_creation], index=["model_blueprint", "model_creation"])
    dataframe.to_csv(nnet_config, header=None, sep=";")
    logger.info("Save nnet_config to {0} done.".format(nnet_config))


def read_nnet_config(nnet_config:str):
    logger.info("Read nnet_config from {0}".format(nnet_config))
    # Use ; sep to avoid some problem in spliting.
    dataframe = pd.read_csv(nnet_config, header=None, index_col=0, sep=";")
    model_blueprint = dataframe.loc["model_blueprint", 1]
    model_creation = dataframe.loc["model_creation", 1]

    return model_blueprint, model_creation


def create_model_dir(model_dir:str, model_blueprint:str, stage=-1):
    # Just change the path of blueprint so that use the copy of blueprint which is in the config directory and it could 
    # avoid unkonw influence from the original blueprint which could be changed possibly before some processes needing 
    # this blueprint, such as pipeline/onestep/extracting_embedings.py
    config_model_blueprint = "{0}/config/{1}".format(
        model_dir, os.path.basename(model_blueprint))

    if not os.path.exists("{0}/log".format(model_dir)):
        os.makedirs("{0}/log".format(model_dir), exist_ok=True)

    if not os.path.exists("{0}/config".format(model_dir)):
        os.makedirs("{0}/config".format(model_dir), exist_ok=True)
    os.makedirs("{0}/checkpoint_info".format(model_dir), exist_ok=True)
    if is_main_training():
        if stage <= 0 and model_blueprint != config_model_blueprint:
            shutil.copy(model_blueprint, config_model_blueprint)
    else:
        while(True):
            if os.path.exists(config_model_blueprint):
                break

    return config_model_blueprint


def draw_list_to_png(list_x, list_y, out_png_file, color='r', marker=None, dpi=256):
    """ Draw a piture for some values.
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(list_x, list_y, color=color, marker=marker)
    plt.savefig(out_png_file, dpi=dpi)
    plt.close()


def read_file_to_list(file_path, every_bytes=10000000):
    list = []
    with open(file_path, 'r') as reader :
            while True :
                lines = reader.readlines(every_bytes)
                if not lines:
                    break
                for line in lines:
                    list.append(line)
    return list


def write_list_to_file(this_list, file_path, mod='w', yml=False):
    """
    @mod: could be 'w' or 'a'
    """
    if not isinstance(this_list,list):
        this_list = [this_list]

    with open(file_path, mod) as writer :
        if yml:
            for x in this_list:
                yaml.dump(x,writer)
        
        else:

            writer.write('\n'.join(str(x) for x in this_list))
            writer.write('\n')


def save_checkpoint(checkpoint_path, **kwargs):
    """Save checkpoint to file for training. Generally, The checkpoint includes
        epoch:<int>
        iter:<int>
        model_path:<string>
        optimizer:<optimizer.state_dict>
        lr_scheduler:<lr_scheduler.state_dict>
    """
    state_dict = {}
    state_dict.update(kwargs)
    torch.save(state_dict, checkpoint_path)


def format(x, str):
    """To hold on the None case when formating float to string.
    @x: a float value or None or any others, should be consistent with str
    @str: a format such as {:.2f}
    """
    if x is None:
        return "-"
    else:
        return str.format(x)


def set_all_seed(seed=None, deterministic=True):
    """This is refered to https://github.com/lonePatient/lookahead_pytorch/blob/master/tools.py.
    """
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = deterministic


def key_to_value(adict, key, return_none=True):
    assert isinstance(adict, dict)

    if key in adict.keys():
        return adict[key]
    elif return_none:
        return None
    else:
        return key


def assign_params_dict(default_params:dict, params:dict, force_check=False, support_unknow=False):

    default_params = copy.deepcopy(default_params)

    default_keys = set(default_params.keys())
    
    # Should keep force_check=False to use support_unknow
    if force_check:
        for key in param.keys():
            if key not in default_keys:
                raise ValueError("The params key {0} is not in default params".format(key))

    # Do default params <= params if they have the same key
    params_keys = set(params.keys())
    for k, v in default_params.items():
        if k in params_keys:
            if isinstance(v, type(params[k])):
                if isinstance(v, dict):
                    # To parse a sub-dict.
                    sub_params = assign_params_dict(v, params[k], force_check, support_unknow)
                    default_params[k] = sub_params
                else:
                    default_params[k] = params[k]
            elif isinstance(v, float) and isinstance(params[k], int):
                default_params[k] = params[k] * 1.0
            elif v is None or params[k] is None:
                default_params[k] = params[k]
            else:
                raise ValueError("The value type of default params [{0}] is "
                "not equal to [{1}] of params for k={2}".format(type(default_params[k]), type(params[k]), k))

    # Support unknow keys
    if not force_check and support_unknow:
        for key in params.keys():
            if key not in default_keys:
                default_params[key] = params[key]

    return default_params


def split_params(params:dict):
    params_split = {"public":{}} 
    params_split_keys = params_split.keys()
    for k, v in params.items():
        if len(k.split(".")) == 2:
            name, param = k.split(".")
            if name in params_split_keys:
                params_split[name][param] = v
            else:
                params_split[name] = {param:v}
        elif len(k.split(".")) == 1:
            params_split["public"][k] = v
        else:
            raise ValueError("Expected only one . in key, but got {0}".format(k))

    return params_split


def auto_str(value, auto=True):
    if isinstance(value, str) and auto:
        return "'{0}'".format(value)
    else:
        return str(value)


def iterator_to_params_str(iterator, sep=",", auto=True):
    return sep.join(auto_str(x, auto) for x in iterator)


def dict_to_params_str(dict, auto=True, connect="=", sep=","):
    params_list = []
    for k, v in dict.items():
        params_list.append(k+connect+auto_str(v, auto))
    return iterator_to_params_str(params_list, sep, False)


def read_log_csv(csv_path:str):
    dataframe = pd.read_csv(csv_path).drop_duplicates(["epoch", "iter"], keep="last", inplace=True)
    return dataframe


### Multi-GPU training [Two solutions: Horovod or DDP]
def init_multi_gpu_training(gpu_id="", solution="ddp"):
    num_gpu = len(parse_gpu_id_option(gpu_id))
    if num_gpu > 1:
        # The DistributedDataParallel (DDP) solution is suggested.
        if solution == "ddp":
            init_ddp()
            if is_main_training(): logger.info("DDP has been initialized.")
        elif solution == "horovod":
            init_horovod()
            if is_main_training(): logger.info("Horovod has been initialized.")
        else:
            raise TypeError("Do not support {} solution for multi-GPU training.".format(method))

def convert_synchronized_batchnorm(model):
    if use_horovod():
        # Synchronize batchnorm for multi-GPU training.
        from .sync_bn import convert_sync_batchnorm
        model = convert_sync_batchnorm(model)
    elif use_ddp():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model

def is_main_training():
    if use_horovod():
        import horovod.torch as hvd
        # Set rank=0 to main training process. See trainer.init_training().
        if hvd.rank() == 0:
            return True
        else:
            return False
    elif use_ddp():
        if dist.get_rank() == 0:
            return True
        else:
            return False
    return True

def auto_scale_lr(lr):
    if use_horovod():
        import horovod.torch as hvd
        return lr * hvd.size()
    elif use_ddp():
        return lr * dist.get_world_size()
    else:
        return lr

## Horovod
def init_horovod():
    os.environ["USE_HOROVOD"] = "true"
    import horovod.torch as hvd
    hvd.init()

def use_horovod():
    return os.getenv("USE_HOROVOD") == "true"

## DDP
def init_ddp():
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("NCCL is not available.")

    # Just plan to support NCCL for GPU-Training with single machine, but it is easy to extend by yourself.
    # Init_method is defaulted to 'env://' (environment) and The IP is 127.0.0.1 (localhost).
    # so do not give these two params to init_process_group.
    # The port will be always defaulted to 29500 by torch that will result in init_process_group failed
    # when number of training task > 1. Use runLauncher.sh, a free port will be auto-setted.
    torch.distributed.init_process_group(backend="nccl")

def use_ddp():
    return torch.distributed.is_initialized()

def cleanup_ddp():
    torch.distributed.destroy_process_group()


# reduce stop signal across ddp progresses, controled by main rank. (Leo 2022-06-20)
def check_exit_ddp(stop_flag=0):
    if use_ddp():
        flag_tensor = torch.zeros(1).to(torch.cuda.current_device())
        if is_main_training():
            flag_tensor +=stop_flag
        dist.all_reduce(flag_tensor,op=dist.ReduceOp.SUM)
        dist.barrier()
        stop_flag = flag_tensor
    return stop_flag


def get_free_port(ip="127.0.0.1"):
    import socket
    # Use contextlib to close socket after return the free port.
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        # Set port as 0, socket will auto-select a free port. And then fetch this port.
        s.bind((ip, 0))
        return s.getsockname()[1]

# https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/utils/data_utils.py
def batch_pad_right(tensors: list, mode="constant", value=0,val_index=-1):
    """Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Parameters
    ----------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.

    """

    if not len(tensors):
        raise IndexError("Tensors list must not be empty")
    # tensors = list(map(list,tensors))

    if len(tensors) == 1:
        # if there is only one tensor in the batch we simply unsqueeze it.
        return tensors[0].unsqueeze(0), torch.tensor([1.0])

    if not (
        any(
            [tensors[i].ndim == tensors[0].ndim for i in range(
                1, len(tensors))]
        )
    ):
        raise IndexError("All tensors must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the last dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(tensors[0].ndim):
        if dim != (tensors[0].ndim - 1):
            if not all(
                [x.shape[dim] == tensors[0].shape[dim] for x in tensors[1:]]
            ):
                raise EnvironmentError(
                    "Tensors should have same dimensions except for last one"
                )
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        # for each tensor we apply pad_right_to
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value
        )
        batched.append(padded)
        valid.append(valid_percent[val_index])

    batched = torch.stack(batched)
    
    return batched, torch.tensor(valid)


def pad_right_to(
    tensor: torch.Tensor, target_shape: (list, tuple), mode="constant", value=0,
):
    """
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Parameters
    ----------
    tensor : input torch tensor
        Input tensor whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode : str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value : float
        Pad value, please refer to torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == tensor.ndim
    # this contains the abs length of the padding for each dimension.
    pads = []
    valid_vals = []  # thic contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1
    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals

class Timer(object):
    def __init__(self) -> None:
        super().__init__()
        self.start_time=time.time()
    def reset(self):
        self.start_time=time.time()
    def elapse(self):
        return time.time()-self.start_time

def csv_to_list(file):
    lists=[]
    with open(file, newline="") as csvfile:
        reader = csv.DictReader(csvfile,delimiter=' ',skipinitialspace=True)
        for rows in reader:
            lists.append(rows)
    return lists



def read_wav_list(wav_file):
    wav_list = []
    with open(wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            item={}
            utt=arr[0]
            wav_path=arr[1]
            item['eg-id']=utt
            item['wav-path']=wav_path
            item['class-label']=0
            wav_list.append(item)
    return  wav_list



def get_torchaudio_backend():
    """Get the backend for torchaudio between soundfile and sox_io according to the os.

    Allow users to use soundfile or sox_io according to their os.

    Returns
    -------
    str
        The torchaudio backend to use.
    """
    current_system = platform.system()
    if current_system == "Windows":
        return "soundfile"
    else:
        return "sox_io"