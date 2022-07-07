# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-11-25
#                              Leo     2022-01-15)
import os
import sys
import re
import logging
import copy
import math
import time
import traceback
import progressbar
import pandas as pd
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from contextlib import nullcontext
from .reporter import Reporter_new as Reporter
from .lr_scheduler_new import LRSchedulerWrapper
from .lr_finder import for_lr_finder_new

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

training_point (this epoch, iter in epoch, global step)
"""

# Trainer ✿


class _BaseTrainer():
    def __init__(self, package, stop_early=False):
        default_elements = {"data": None, "model": None,
                            "optimizer": None, "lr_scheduler": None}
        default_params = {"model_dir": "", "model_blueprint": "", "exist_model": "", "start_epoch": 0, "epochs": 10,
                          "use_gpu": True, "gpu_id": "", "benchmark": True, "max_change": 10.0, "use_amp": False, "accum_grad": 1,
                          "compute_accuracy": True, "compute_valid_accuracy": True, "compute_batch_num_valid": 1,
                          "suffix": "params", "nan_debug": False, "skip_nan_batch": True, "use_tensorboard": True}

        elements, params = package
        self.elements = utils.assign_params_dict(default_elements, elements)
        self.params = utils.assign_params_dict(
            default_params, params, support_unknow=True)

        assert self.elements["data"] is not None
        assert self.elements["model"] is not None
        assert self.elements["optimizer"] is not None

        assert self.params["model_dir"] != ""
        assert self.params["model_blueprint"] != ""

        self.elements["model_forward"] = self.elements["model"]
        self.params["start_epoch"] = max(0, self.params["start_epoch"])
        self.params["accum_grad"] = max(1,self.params["accum_grad"])
        self.use_ddp = utils.use_ddp()
        self.stop_early = stop_early  # To do.

        # Automatic mixed precision init.(Leo 2021-11-08)
        self.scaler = torch.cuda.amp.GradScaler() if self.params["use_amp"] else None

        # (epoch, iter in epoch, global step)
        self.training_point = copy.deepcopy([self.params["start_epoch"], 0, 0])
        self.cycle_point = 0  # for cycle training.

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
            utils.write_nnet_config(
                model_blueprint, model_creation, "{0}/config/nnet.config".format(model_dir))

        # Recover checkpoint | Tansform learning | Initialize parametes
        if start_epoch > 0:
            # This train_stage is equal to number of completed epoch
            if utils.is_main_training():
                logger.info(
                    "Recover training from {0} epoch.".format(start_epoch))
            model.load_state_dict(torch.load('{0}/{1}.{2}'.format(model_dir, start_epoch, suffix),
                                             map_location="cpu"))
            # info_path = '{0}/{1}/{2}.{3}'.format(
            #     model_dir, "checkpoint_info", start_epoch, "info")
            info_log_path = '{0}/{1}/{2}.{3}'.format(
                model_dir, "checkpoint_info", start_epoch, "yaml")
            if os.path.exists(info_log_path):
                # info = torch.load(info_path)
                # self.elements["optimizer"].load_state_dict(info['optimizer'])
                # for state in self.elements["optimizer"].values():
                #     for k, v in state.items():
                #         if isinstance(v, torch.Tensor):
                #             state[k] = v.cuda()
                with open(info_log_path, 'r') as fin:
                    info = yaml.load(fin, Loader=yaml.FullLoader)
                self.training_point[2] = info['step']
        elif os.path.exists(exist_model):
            if utils.is_main_training():
                logger.info(
                    "Use {0} as the initial model to start transform-training.".format(exist_model))
            model.load_transform_state_dict(
                torch.load(exist_model, map_location="cpu"))
        else:
            # Just use the raw initial model or initialize it again by some initial functions here
            pass  # Now, it means use the raw initial model

        # Select device

        model = self.select_device()

        # Original model is built in libs.nnet.framework.TopVirtualNnet, and it is not available after
        # wrapped by DistributedDataParallel. So, to call functions of TopVirtualNnet conveniently, the
        # self.elements["model_forward"] is set here to name DistributedDataParallel.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):

            self.elements["model"] = model.module
            self.elements["model_forward"] = model


    def save_model(self, mod="epoch",train_lr=None,valid_loss=None):
        assert mod in ["epoch", "iter", "cycle"]
        if mod == "epoch":
            model_name = self.training_point[0]
        elif mod == "iter":
            model_name = "{}.{}".format(
                self.training_point[0], self.training_point[1])
        else:
            model_name = "{}_cycle".format(self.cycle_point)
        model_path = '{0}/{1}.{2}'.format(
            self.params["model_dir"], model_name, self.params["suffix"])

        # info = {
        #     'optimizer': self.elements["optimizer"].state_dict(),
        #     'step': self.training_point[2],
        # }
        info_log = {
            'train_lr': train_lr if train_lr else "see train.csv",
            "next_lr": self.elements["optimizer"].state_dict()['param_groups'][0]['lr'],
            'epoch': self.training_point[0],
            'iter in epoch': self.training_point[1],
            'step': self.training_point[2],
            'valid_loss':valid_loss if valid_loss else "see train.csv"
        }
        # info_path = '{0}/{1}/{2}.{3}'.format(
        #     self.params["model_dir"], "checkpoint_info", model_name, "info")
        # info_log_path = re.sub('.info$', '.yaml', info_path)
        info_log_path = '{0}/{1}/{2}.{3}'.format(
            self.params["model_dir"], "checkpoint_info", model_name, "yaml")
        logger.info("Save model to {0}. \n epoch/iter: {1}/{2}.  cur_step: {3}".format(model_path, self.training_point[0],
                                                                                       self.training_point[1], self.training_point[2]))
        torch.save(self.elements["model"].state_dict(), model_path)
        # torch.save(info, info_path)
        with open(info_log_path, 'w') as fout:
            data = yaml.dump(info_log)
            fout.write(data)

    def run(self):
        raise NotImplementedError

    @for_lr_finder_new
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
        self.num_batch=0
    def train_one_batch(self, batch, step_lr=True):
        """A normal training core without fetching data from iterator.
        """
        model = self.elements["model"]
        model_forward = self.elements["model_forward"]
        optimizer = self.elements["optimizer"]

        if not model.training:
            model.train()

        if self.params["nan_debug"]:
            device = utils.get_device(self.elements["model"])
            inputs = torch.load(
                "{0}/nan.batch".format(self.params["model_dir"])).to(device)
            targets = torch.load(
                "{0}/nan.targets".format(self.params["model_dir"])).to(device)
            self.elements["model"].load_state_dict(torch.load("{0}/nan.params".format(self.params["model_dir"]),
                                                              map_location="cpu"))
            self.elements["model"].to(device)
        else:
            inputs, targets = batch

        context = None
        # Disable gradient synchronizations across DDP processes.
        # Within this context, gradients will be accumulated on module
        # variables, which will later be synchronized.
        if self.use_ddp and self.num_batch%self.params["accum_grad"]!=0:
            context = model_forward.no_sync

        # Used for single gpu training and DDP gradient synchronization
        # processes.
        else:
            context = nullcontext
        with context():
            # Managing automatic mixed precision  (Leo 2021-11-08)
            with torch.cuda.amp.autocast(self.scaler is not None):

                loss = model.get_loss(model_forward(inputs), targets)/self.params["accum_grad"]
                # loss = model_forward(inputs, targets)/self.params["accum_grad"]

            if self.params["use_amp"]:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        loss.detach()  # For safe.
        loss = loss.item()*self.params["accum_grad"]
        accuracy = None


        if self.num_batch%self.params["accum_grad"]==0:
            # Use mixed precision training (Leo 2021-11-08)
            if self.params["use_amp"]:
                self.scaler.unscale_(optimizer)
                if not self.modify_grad() and not self.params["skip_nan_batch"]:
                    torch.save(inputs.cpu(), "{0}/nan.batch".format(self.params["model_dir"]))
                    torch.save(targets.cpu(), "{0}/nan.targets".format(self.params["model_dir"]))
                    torch.save(self.elements["model"].state_dict(), "{0}/nan.params".format(self.params["model_dir"]))
                    raise RuntimeError('There is Nan problem in epoch/iter: {0}/{1} (nan batch and params are saved in {2})'.format(self.training_point[0],
                                                                                                                                    self.training_point[1], "{0}/nan.*".format(self.params["model_dir"])))
                else:
                    self.scaler.step(optimizer)
                self.scaler.update()
            else:
                if not self.modify_grad():
                    if not self.params["skip_nan_batch"]:
                        torch.save(inputs.cpu(), "{0}/nan.batch".format(self.params["model_dir"]))
                        torch.save(targets.cpu(), "{0}/nan.targets".format(self.params["model_dir"]))
                        torch.save(self.elements["model"].state_dict(), "{0}/nan.params".format(self.params["model_dir"]))
                        raise RuntimeError('There is Nan problem in epoch/iter: {0}/{1} (nan batch and params are saved in {2})'.format(self.training_point[0],
                                                                                                                                        self.training_point[1], "{0}/nan.*".format(self.params["model_dir"])))
                else:
                    optimizer.step()

            optimizer.zero_grad()
            self.training_point[2] += 1 # update step
            accuracy = model.get_accuracy(targets) if self.params["compute_accuracy"] else None
            if step_lr:
                self.step_lr(loss,accuracy,optimizer,self.elements["lr_scheduler"])
        self.num_batch += 1

        return loss, accuracy

    # clip grad
    def modify_grad(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.elements["model"].parameters(), self.params["max_change"])
        if not torch.isfinite(grad_norm):
            logger.warning("Grad:{0} is not finite in epoch/iter: {1}/{2}".format(grad_norm, self.training_point[0],self.training_point[1]))
            if self.params["nan_debug"]:
                raise RuntimeError(
                    "[NOT OK] Nan is still found in this debug.")
            return False
        else:
            if self.params["nan_debug"]:
                raise RuntimeError(
                    "[OK] There is no nan found for this debug.")
            return True

    def compute_validation(self, data_loader):
        """A normal evaluation core.
        """
        model = self.elements["model"]
        model_forward = self.elements["model_forward"]
        train_status = model.training  # Record status.
        model.eval()

        loss = 0.
        accuracy = 0. if self.params["compute_valid_accuracy"] else None
        num_samples = 0
        with torch.no_grad():
            for idx,this_data in enumerate(data_loader):
                inputs, targets = this_data
                num_utts = targets.size(0)
                if num_utts == 0:
                    continue
                # in valid stage, DO NOT call ddp model, for ddp model is in JOIN context wrapper.
                # Leo 2022-02-03
                loss += model.get_loss(model(inputs),
                                       targets).item() * len(targets)

                # loss += model_forward(inputs,targets).item() * len(targets)
                num_samples += len(targets)

                if self.params["compute_valid_accuracy"]:
                    # This will occupy extra GPU memory.
                    accuracy += model.get_accuracy(targets) * len(targets)
                if idx > self.params["compute_batch_num_valid"]-1:
                    break
            avg_loss = loss/num_samples
            avg_accuracy = accuracy / num_samples if self.params["compute_valid_accuracy"] else None
        if train_status:
            model.train()

        return avg_loss, avg_accuracy

    def step_lr(self,train_loss,train_acc,base_optimizer,lr_scheduler):

        # For multi-GPU training. Remember that it is not convenient to wrap lr_scheduler
        # for there are many strategies with different details. Here, only warmR, ReduceLROnPlateau
        # and some simple schedulers whose step() parameter is 'epoch' only are supported.

        valid_dataloader=self.elements["data"].valid_loader

        lr_scheduler_params = {
            "training_point": self.training_point}
        valid_loss = None
        valid_computed = False
        if lr_scheduler.name == "reduceP" and lr_scheduler.is_reduce_point(self.training_point):
            assert valid_dataloader is not None
            valid_loss, valid_acc = self.compute_validation(valid_dataloader)
            lr_scheduler_params["valid_metric"] = (valid_loss, valid_acc)
            valid_computed = True
        if utils.is_main_training():
            if valid_computed or (valid_dataloader is not None and self.reporter.is_report(self.training_point)):
                if not valid_computed:
                    valid_loss, valid_acc = self.compute_validation(valid_dataloader)

                    valid_computed = False
                # real_snapshot is set for tensorboard to avoid workspace problem
                real_snapshot = {"train_loss": train_loss, "valid_loss": valid_loss,
                                "train_acc": train_acc*100, "valid_acc": valid_acc*100}
                snapshot = {"train_loss": "{0:.6f}".format(train_loss), "valid_loss": "{0:.6f}".format(valid_loss),
                            "train_acc": "{0:.2f}".format(train_acc*100), "valid_acc": "{0:.2f}".format(valid_acc*100),
                            "total_dur":self.origin_total_dur,"num_sample":self.num_sample,"real": real_snapshot}
                # For ReduceLROnPlateau.
                lr_scheduler_params["valid_metric"] = (valid_loss, valid_acc)
            else:
                real_snapshot = {
                    "train_loss": train_loss, "train_acc": train_acc*100}
                snapshot = {"train_loss": "{0:.6f}".format(train_loss), "valid_loss": "",
                            "train_acc": "{0:.2f}".format(train_acc*100), "valid_acc": "",
                            "total_dur":self.origin_total_dur,"num_sample":self.num_sample,"real": real_snapshot}
            training_point = (self.training_point[0],self.training_point[1],self.training_point[2])
            self.train_lr = base_optimizer.state_dict()['param_groups'][0]['lr']
            self.reporter.update(snapshot,training_point,self.train_lr)
        if lr_scheduler is not None:
            # It is not convenient to wrap lr_scheduler (doing).
            if isinstance(lr_scheduler, LRSchedulerWrapper):
                lr_scheduler.step(**lr_scheduler_params)
                if utils.is_main_training():
                    current_lr = base_optimizer.state_dict()['param_groups'][0]['lr']
                    if lr_scheduler.name == "reduceP":
                        if current_lr < self.last_lr:
                            self.last_lr = current_lr
                            self.save_model(mod="iter",train_lr=self.train_lr,valid_loss=valid_loss)
                        elif current_lr <= lr_scheduler.min_lr and lr_scheduler.is_reduce_point(self.training_point):
                            self.save_model(mod="iter",train_lr=self.train_lr,valid_loss=valid_loss)

                    if lr_scheduler.is_cycle_point(self.training_point):
                        self.cycle_point+=1
                        self.save_model(mod="cycle",train_lr=self.train_lr,valid_loss=valid_loss)
            else:
                # For some pytorch lr_schedulers, but it is not available for all.
                lr_scheduler.step(self.training_point[0])


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
            # See init_training.
            model_forward = self.elements["model_forward"]
            self.train_lr = self.elements["optimizer"].state_dict()['param_groups'][0]['lr']
            self.last_lr =  self.elements["optimizer"].state_dict()['param_groups'][0]['lr']

            if utils.is_main_training():
                logger.info("Training will run for {0} epochs.".format(epochs))
            if utils.is_main_training() and self.params["accum_grad"] > 1:
                logger.info("using accumulate grad,accum_num: {}".format(
                    self.params["accum_grad"]))

            if isinstance(model_forward, torch.nn.parallel.DistributedDataParallel):

                model_context = model_forward.join(throw_on_early_termination=True)
            else:
                model_context = nullcontext()
            with model_context:
                for this_epoch in range(start_epoch, epochs):
                    self.training_point[0]+=1

                    data.train_loader.dataset.set_epoch(this_epoch)

                    # with model_context:

                    for _, batch in enumerate(data.train_loader, 0):
                        # It is important for reporter.
                        if utils.is_main_training() and self.training_point[1]==0: self.origin_total_dur,self.num_sample=data.train_loader.dataset.get_data_dur()

                        self.training_point[1] +=1

                        num_utts = batch[0].size(0)

                        if num_utts == 0:
                            continue
                        if model.use_step:
                            step_point = (self.training_point[0],self.training_point[2])
                            model.step_iter(*step_point)

                        loss, acc = self.train_one_batch(batch)

                        model.backward_step(*self.training_point)

                    if utils.is_main_training():self.save_model(train_lr=self.train_lr)
                    self.training_point[1] =0
            if utils.is_main_training():self.reporter.finish()
            if utils.is_main_training():
                final_model_name = "{}_cycle".format(self.cycle_point) if self.cycle_point else epochs
                final_model_path = os.path.join(self.params["model_dir"],'final.params')
                if os.path.exists(final_model_path) or os.path.islink(final_model_path):
                    os.remove(final_model_path)

                os.symlink('{0}/{1}.{2}'.format(self.params["model_dir"], final_model_name, self.params["suffix"]), final_model_path)
        except BaseException as e:
            if utils.use_ddp():utils.cleanup_ddp()
            if not isinstance(e, KeyboardInterrupt):traceback.print_exc()
            sys.exit(1)

    @for_lr_finder_new
    def lr_finder_compute(self, train_batch):
        model = self.elements["model"]
        if model.use_step:
            step_point = (self.training_point[0],self.training_point[2])
            model.step_iter(*step_point)
        loss, acc = self.train_one_batch(train_batch,step_lr=False)
        model.backward_step(*self.training_point)
        valid_loss, valid_acc=0,0
        if utils.is_main_training():
            valid_loss, valid_acc = self.compute_validation(
                self.elements["data"].valid_loader)
        return ["train_loss", "train_acc", "valid_loss", "valid_acc"], [loss, acc, valid_loss, valid_acc]

    def run_lr_finder(self, save_file: str, comment=None, init_lr=1e-8, final_lr=10., num_iters=2000, beta=0.98):
        self.init_training()
        log_dir = self.params["model_dir"] + "/log/"  # For tensorboardX
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
        if utils.use_ddp():utils.cleanup_ddp()
        sys.exit(1)

class MultitaskTrainer(_BaseTrainer):
    """One input and multi-output corresponding to different tasks, such as one 
    is speaker classfication and another is phones classfication.
    """
    pass


class GANTrainer(_BaseTrainer):
    """This is for GAN.
    """
    pass


# Function ✿
def add_gaussian_noise_to_grad(model, t, n=0.1, gamma=0.55):
    """ADDING GRADIENT NOISE IMPROVES LEARNING FOR VERY DEEP NETWORKS.
    """
    var = n/(1+t)**gamma
    for param in model.params():
        param.grad += to_device(model, torch.normal(0, var, param.size()))
