# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-11-25)

import os, sys
import logging
import math
import time
import traceback
import progressbar
import pandas as pd
import numpy as np
import torch

from .reporter import Reporter
from .lr_scheduler import LRSchedulerWrapper
from .lr_finder import for_lr_finder

import libs.support.utils as utils

# Wrap stderr before logger init.
progressbar.streams.wrap_stderr()

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

"""
This is the structure of Package:

Package(
    Elements{
        data:Bunch
        model:TopVirtualNnet
        optimizer:Optimizer
        lr_scheduler:LR_Scheduler
        },

    Params{
        model_dir:str
        exist_model:str
        start_epoch:int
        epochs:int
        ...
        }
    )

training_point(this_epoch, this_iter, data.num_batch_train)
"""

## Trainer ✿
class _BaseTrainer():
    def __init__(self, package, stop_early=False):
        default_elements = {"data":None, "model":None, "optimizer":None, "lr_scheduler":None}
        default_params = {"model_dir":"", "model_blueprint":"", "exist_model":"", "start_epoch":0, "epochs":10, 
                          "use_gpu":True, "gpu_id":"", "benchmark":True, "max_change":10.0, 
                          "compute_accuracy":True, "compute_valid_accuracy":True, "compute_one_batch_valid":True,
                          "suffix":"params", "nan_debug":False, "use_tensorboard":True, "ddp_random_epoch":False}

        elements, params = package
        self.elements = utils.assign_params_dict(default_elements, elements)
        self.params = utils.assign_params_dict(default_params, params, support_unknow=True)

        assert self.elements["data"] is not None
        assert self.elements["model"] is not None
        assert self.elements["optimizer"] is not None

        assert self.params["model_dir"] != ""
        assert self.params["model_blueprint"] != ""

        self.elements["model_forward"] = self.elements["model"]
        self.params["start_epoch"] = max(0, self.params["start_epoch"])

        self.stop_early = stop_early # To do.
        self.training_point = (self.params["start_epoch"], 0, self.elements["data"].num_batch_train)

    def select_device(self):
        return utils.select_model_device(self.elements["model"], self.params["use_gpu"], 
                                          gpu_id=self.params["gpu_id"], benchmark=self.params["benchmark"])

    def init_training(self):
        model = self.elements["model"]
        start_epoch = self.params["start_epoch"]
        exist_model = self.params["exist_model"]
        model_dir = self.params["model_dir"]
        model_blueprint = self.params["model_blueprint"]
        suffix = self.params["suffix"]

        if start_epoch <= 0 and utils.is_main_training():
            model_creation = model.get_model_creation()
            utils.write_nnet_config(model_blueprint, model_creation, "{0}/config/nnet.config".format(model_dir))

        ## Recover checkpoint | Tansform learning | Initialize parametes 
        if start_epoch > 0:
            # This train_stage is equal to number of completed epoch
            if utils.is_main_training(): logger.info("Recover training from {0} epoch.".format(start_epoch))
            model.load_state_dict(torch.load('{0}/{1}.{2}'.format(model_dir, start_epoch, suffix), 
                                             map_location="cpu"))
        elif os.path.exists(exist_model):
            if utils.is_main_training(): logger.info("Use {0} as the initial model to start transform-training.".format(exist_model))
            model.load_transform_state_dict(torch.load(exist_model, map_location="cpu"))
        else:
            # Just use the raw initial model or initialize it again by some initial functions here
            pass # Now, it means use the raw initial model

        if utils.use_horovod():
            import horovod.torch as hvd

            # Broadcast parameters from rank 0 to all other processes.
            hvd.broadcast_parameters(self.elements["model"].state_dict(), root_rank=0)

             # For optimizer wrapper such as lookahead.
            if getattr(self.elements["optimizer"], "optimizer", None) is not None:
                raise TypeError("Do not support using lookahead with horovod now.")
            else:
                # Broadcast optimizer state.
                hvd.broadcast_optimizer_state(self.elements["optimizer"], root_rank=0)
                self.elements["optimizer"] = hvd.DistributedOptimizer(self.elements["optimizer"], 
                                             named_parameters=self.elements["model"].named_parameters())

        ## Select device
        model = self.select_device()

        # Original model is built in libs.nnet.framework.TopVirtualNnet, and it is not available after
        # wrapped by DistributedDataParallel. So, to call functions of TopVirtualNnet conveniently, the 
        # self.elements["model_forward"] is set here to name DistributedDataParallel.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.elements["model"] = model.module
            self.elements["model_forward"] = model

    def save_model(self, from_epoch=True):
        if from_epoch:
            model_name = self.training_point[0]+1
        else:
            model_name = "{}.{}".format(self.training_point[0]+1, self.training_point[1]+1)

        model_path = '{0}/{1}.{2}'.format(self.params["model_dir"], model_name, self.params["suffix"])
        logger.info("Save model from {0}/{1} of {2} epoch to {3}.".format(self.training_point[1]+1, self.training_point[2], 
                                                                 self.training_point[0]+1, model_path))
        torch.save(self.elements["model"].state_dict(), model_path)

    def run(self):
        raise NotImplementedError

    @for_lr_finder
    def lr_finder_compute(self, train_batch):
        # Only train_batch parameter for it's always main metric.
        raise NotImplementedError

    def run_lr_finder(self):
        # Implement this function w.r.t self.lr_finder_compute().
        raise NotImplementedError


class SimpleTrainer(_BaseTrainer):
    """One input and one output.
    """
    def __init__(self, *args, **kwargs):
        super(SimpleTrainer, self).__init__(*args, **kwargs)

    def train_one_batch(self, batch):
        """A normal training core without fetching data from iterator.
        """
        model = self.elements["model"]
        model_forward = self.elements["model_forward"]
        optimizer = self.elements["optimizer"]

        if not model.training:
            model.train()

        if self.params["nan_debug"]:
            device = utils.get_device(self.elements["model"])
            inputs = torch.load("{0}/nan.batch".format(self.params["model_dir"])).to(device)
            targets = torch.load("{0}/nan.targets".format(self.params["model_dir"])).to(device)
            self.elements["model"].load_state_dict(torch.load("{0}/nan.params".format(self.params["model_dir"]), 
                                             map_location="cpu"))
            self.elements["model"].to(device)
        else:
            inputs, targets = batch
        optimizer.zero_grad()

        loss = model.get_loss(model_forward(inputs), targets)
        loss.backward()
        loss.detach() # For safe.

        if self.params["max_change"] > 0:
            # Reference:https://github.com/horovod/horovod/blob/master/horovod/torch/__init__.py:420~423.
            # Synchronize the grad for grad_norm when using horovod.
            if utils.use_horovod(): optimizer.synchronize()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.params["max_change"])

            if math.isnan(grad_norm):
                if self.params["nan_debug"]:
                    raise RuntimeError("[NOT OK] Nan is still found in this debug.")
                torch.save(inputs.cpu(), "{0}/nan.batch".format(self.params["model_dir"]))
                torch.save(targets.cpu(), "{0}/nan.targets".format(self.params["model_dir"]))
                torch.save(self.elements["model"].state_dict(), "{0}/nan.params".format(self.params["model_dir"]))
                raise RuntimeError('There is Nan problem in iter/epoch: {0}/{1} (nan batch and params are saved in {2})'.format(self.training_point[1]+1, 
                self.training_point[0]+1, "{0}/nan.*".format(self.params["model_dir"])))
            else:
                if self.params["nan_debug"]:
                    raise RuntimeError("[OK] There is no nan found for this debug.")
                if utils.use_horovod():
                    with optimizer.skip_synchronize():
                        optimizer.step()
                else:
                    optimizer.step()
        else:
            optimizer.step()

        accuracy = model.get_accuracy(targets) if self.params["compute_accuracy"] else None

        return loss.item(), accuracy

    def compute_validation(self, data_loader):
        """A normal evaluation core.
        """
        model = self.elements["model"]
        model_forward = self.elements["model_forward"]
        train_status = model.training # Record status.
        model.eval()

        loss = 0.
        accuracy = 0. if self.params["compute_valid_accuracy"] else None

        num_samples = 0
        with torch.no_grad():
            for this_data in data_loader:
                inputs, targets = this_data
                loss += model.get_loss(model_forward(inputs), targets).item() * len(targets)
                num_samples += len(targets)

                if self.params["compute_valid_accuracy"]:
                    # This will occupy extra GPU memory.
                    accuracy += model.get_accuracy(targets) * len(targets)

                if self.params["compute_one_batch_valid"]:
                    break

            avg_loss = loss/num_samples
            avg_accuracy = accuracy/num_samples if self.params["compute_valid_accuracy"] else None

        if train_status:
            model.train()

        return avg_loss, avg_accuracy

    def run(self):
        """Main function to start a training process.
        """
        try:
            self.init_training()

            if utils.is_main_training():
                self.reporter = Reporter(self)

            start_epoch = self.params["start_epoch"]
            epochs = self.params["epochs"]
            data = self.elements["data"]
            model = self.elements["model"]
            model_forward = self.elements["model_forward"] # See init_training.
            lr_scheduler = self.elements["lr_scheduler"]
            base_optimizer = self.elements["optimizer"]

            # For lookahead.
            if getattr(base_optimizer, "optimizer", None) is not None:
                base_optimizer = base_optimizer.optimizer
            last_lr = base_optimizer.state_dict()['param_groups'][0]['lr']

            if utils.is_main_training(): logger.info("Training will run for {0} epochs.".format(epochs))

            for this_epoch in range(start_epoch, epochs):
                # Set random seed w.r.t epoch for distributed training.
                if isinstance(data.train_loader.sampler, torch.utils.data.distributed.DistributedSampler) and \
                    self.params["ddp_random_epoch"]:
                    data.train_loader.sampler.set_epoch(this_epoch)
                for this_iter, batch in enumerate(data.train_loader, 0):
                    self.training_point = (this_epoch, this_iter, data.num_batch_train) # It is important for reporter.

                    if model.use_step:
                        model.step(*self.training_point)

                    loss, acc = self.train_one_batch(batch)

                    model.backward_step(*self.training_point)

                    # For multi-GPU training. Remember that it is not convenient to wrap lr_scheduler 
                    # for there are many strategies with different details. Here, only warmR, ReduceLROnPlateau
                    # and some simple schedulers whose step() parameter is 'epoch' only are supported. 
                    lr_scheduler_params = {"training_point":self.training_point}

                    valid_computed = False
                    if lr_scheduler.name == "reduceP" and lr_scheduler.is_reduce_point(self.training_point):
                        assert data.valid_loader is not None
                        valid_loss, valid_acc = self.compute_validation(data.valid_loader)
                        lr_scheduler_params["valid_metric"] = (valid_loss, valid_acc)
                        valid_computed = True

                    if utils.is_main_training():
                        if valid_computed or (data.valid_loader and self.reporter.is_report(self.training_point)):
                            if not valid_computed:
                                valid_loss, valid_acc = self.compute_validation(data.valid_loader)
                                valid_computed = False

                            # real_snapshot is set for tensorboard to avoid workspace problem
                            real_snapshot = {"train_loss":loss, "valid_loss":valid_loss, 
                                             "train_acc":acc*100, "valid_acc":valid_acc*100}
                            snapshot = {"train_loss":"{0:.6f}".format(loss), "valid_loss":"{0:.6f}".format(valid_loss), 
                                        "train_acc":"{0:.2f}".format(acc*100), "valid_acc":"{0:.2f}".format(valid_acc*100),
                                        "real":real_snapshot}
                            # For ReduceLROnPlateau.
                            lr_scheduler_params["valid_metric"] = (valid_loss, valid_acc)
                        else:
                            real_snapshot = {"train_loss":loss, "train_acc":acc*100}
                            snapshot = {"train_loss":"{0:.6f}".format(loss), "valid_loss":"",
                                        "train_acc":"{0:.2f}".format(acc*100), "valid_acc":"",
                                        "real":real_snapshot}

                    if lr_scheduler is not None:
                        # It is not convenient to wrap lr_scheduler (doing).
                        if isinstance(lr_scheduler, LRSchedulerWrapper):
                            lr_scheduler.step(**lr_scheduler_params)
                            if utils.is_main_training():
                                current_lr = base_optimizer.state_dict()['param_groups'][0]['lr']
                                if lr_scheduler.name == "reduceP":
                                    if current_lr < last_lr:
                                        last_lr = current_lr
                                        self.save_model(from_epoch=False)
                                    elif current_lr <= lr_scheduler.min_lr and lr_scheduler.is_reduce_point(self.training_point):
                                        self.save_model(from_epoch=False)
                        else:
                            # For some pytorch lr_schedulers, but it is not available for all.
                            lr_scheduler.step(this_epoch)
                    if utils.is_main_training(): self.reporter.update(snapshot)
                if utils.is_main_training(): self.save_model()
            if utils.is_main_training(): self.reporter.finish()
        except BaseException as e:
                if utils.use_ddp(): utils.cleanup_ddp()
                if not isinstance(e, KeyboardInterrupt):
                    traceback.print_exc()
                sys.exit(1) 

    @for_lr_finder
    def lr_finder_compute(self, train_batch):
        model = self.elements["model"]
        if model.use_step:
            model.step(*self.training_point)
        loss, acc = self.train_one_batch(train_batch)
        model.backward_step(*self.training_point)
        if utils.is_main_training():
            valid_loss, valid_acc = self.compute_validation(self.elements["data"].valid_loader)
        return ["train_loss", "train_acc", "valid_loss", "valid_acc"], [loss, acc, valid_loss, valid_acc]

    def run_lr_finder(self, save_file:str, comment=None, init_lr=1e-8, final_lr=10., num_iters=None, beta=0.98):
        self.init_training()
        log_dir =  self.params["model_dir"] + "/log/" # For tensorboardX
        if comment is not None:
            save_file = comment + "-" + save_file
        save_file = log_dir + save_file
        log_lrs, values_matrix = self.lr_finder_compute(self.elements["data"].train_loader, self.elements["optimizer"], 
                                                        init_lr=init_lr, final_lr=final_lr, num_iters=num_iters, beta=beta, 
                                                        log_dir=log_dir, comment=comment)
        if utils.is_main_training():
            df = pd.DataFrame(np.vstack([log_lrs, values_matrix]).T, 
                            columns=["log_lr", "train_loss", "train_acc", "valid_loss", "valid_acc"])
            logger.info("Save lr finder values to {}.".format(save_file))
            df.to_csv(save_file)


class MultitaskTrainer(_BaseTrainer):
    """One input and multi-output corresponding to different tasks, such as one 
    is speaker classfication and another is phones classfication.
    """
    pass



class GANTrainer(_BaseTrainer):
    """This is for GAN.
    """
    pass



## Function ✿
def add_gaussian_noise_to_grad(model, t, n=0.1, gamma=0.55):
    """ADDING GRADIENT NOISE IMPROVES LEARNING FOR VERY DEEP NETWORKS.
    """
    var=n/(1+t)**gamma
    for param in model.params():
        param.grad += to_device(model, torch.normal(0, var, param.size()))
