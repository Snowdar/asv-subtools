# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-09)

import os, sys
import time
import shutil
import logging
import progressbar
import traceback
import pandas as pd

from multiprocessing import Process, Queue

import libs.support.utils as utils

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Reporter():
    def __init__(self, trainer):
        default_params = {
            "report_times_every_epoch":None,
            "report_interval_iters":100,
            "record_file":"train.csv"
        }
        self.trainer = trainer
        default_params = utils.assign_params_dict(default_params, self.trainer.params)
        
        if default_params["report_times_every_epoch"] is not None:
            self.report_interval_iters = max(1, self.trainer.training_point[2]//default_params["report_times_every_epoch"])
        else:
            self.report_interval_iters = default_params["report_interval_iters"]

        self.epochs = self.trainer.params["epochs"]

        self.optimizer = self.trainer.elements["optimizer"]

        # For optimizer wrapper such as lookahead.
        if getattr(self.optimizer, "optimizer", None) is not None:
            self.optimizer = self.optimizer.optimizer

        self.device = "[{0}]".format(utils.get_device(self.trainer.elements["model"]))

        self.record_value = []

        self.start_write_log = False
        if default_params["record_file"] != "" and default_params["record_file"] is not None:
            self.record_file = "{0}/log/{1}".format(self.trainer.params["model_dir"], default_params["record_file"])

            # The case to recover training
            if self.trainer.params["start_epoch"] > 0:
                self.start_write_log = True
            elif os.path.exists(self.record_file):
                # Do backup to avoid clearing the loss log when re-running a same launcher.
                bk_file = "{0}.bk.{1}".format(self.record_file, time.strftime('%Y_%m_%d.%H_%M_%S',time.localtime(time.time())))
                shutil.move(self.record_file, bk_file)
        else:
            self.record_file = None

        # A format to show progress
        # Do not use progressbar.Bar(marker="\x1b[32m█\x1b[39m") and progressbar.SimpleProgress(format='%(value_s)s/%(max_value_s)s') to avoid too long string.
        widgets=[progressbar.Percentage(format='%(percentage)3.2f%%'), " | ",
                 "Epoch:", progressbar.Variable('current_epoch', format='{formatted_value}', width=0, precision=0), "/{0}, ".format(self.epochs),
                 "Iter:", progressbar.Variable('current_iter', format='{formatted_value}', width=0, precision=0), "/{0}".format(self.trainer.training_point[2]),
                 " (", progressbar.Timer(format='ELA: %(elapsed)s'), ", ",progressbar.AdaptiveETA(), ")"]

        max_value = self.trainer.params["epochs"]*self.trainer.training_point[2]

        self.bar = progressbar.ProgressBar(max_value=max_value, widgets=widgets, redirect_stdout=True)

        # Use multi-process for update.
        self.queue = Queue()
        self.process = Process(target=self._update, daemon=True)
        self.process.start()

    def is_report(self, training_point):
        return (training_point[1]%self.report_interval_iters == 0 or \
                training_point[1] + 1 == training_point[2])

    def record(self, info_dict, training_point):
        if self.record_file is not None:
            self.record_value.append(info_dict)

            if self.is_report(training_point):
                print("Device:{0}, {1}".format(self.device, utils.dict_to_params_str(info_dict, auto=False, sep=", ")))
                dataframe = pd.DataFrame(self.record_value)
                if self.start_write_log:
                    dataframe.to_csv(self.record_file, mode='a', header=False, index=False)
                else:
                    # with open(self.record_file, "w") as f:
                    #     f.truncate()
                    dataframe.to_csv(self.record_file, header=True, index=False)
                    self.start_write_log = True
                self.record_value.clear()

    def _update(self):
        # Do not use any var which will be updated by main process, such as self.trainer.training_point.
        while True:
            try:
                res = self.queue.get()
                if res is None:
                    self.bar.finish()
                    break

                snapshot, training_point, current_lr = res
                current_epoch, current_iter, num_batchs_train = training_point
                update_iters = current_epoch * num_batchs_train + current_iter + 1
                self.bar.update(update_iters, current_epoch=current_epoch+1, current_iter=current_iter+1)

                info_dict = {"epoch":current_epoch+1, "iter":current_iter+1, "position":update_iters, 
                             "lr":"{0:.8f}".format(current_lr)}
                info_dict.update(snapshot)
                self.record(info_dict, training_point)
            except BaseException as e:
                self.bar.finish()
                if not isinstance(e, KeyboardInterrupt):
                    traceback.print_exc()
                sys.exit(1)

    def update(self, snapshot:dict):
        # One update calling and one using of self.trainer.training_point and current_lr.
        current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.queue.put((snapshot, self.trainer.training_point, current_lr))

    def finish(self):
        self.queue.put(None)
        # Wait process completed.
        self.process.join()


class LRFinderReporter():
    def __init__(self, max_value):
        widgets=[progressbar.Percentage(format='%(percentage)3.2f%%'), " | ", "Iter:",
                 progressbar.Variable('current_iter', format='{formatted_value}', width=0, precision=0), "/{0}".format(max_value), ", ",
                 progressbar.Variable('snapshot', format='{formatted_value}', width=8, precision=0),
                 " (", progressbar.Timer(format='ELA: %(elapsed)s'), ", ",progressbar.AdaptiveETA(), ")"]

        self.bar = progressbar.ProgressBar(max_value=max_value, widgets=widgets, redirect_stdout=True)

        # Use multi-process for update.
        self.queue = Queue()
        self.process = Process(target=self._update, daemon=True)
        self.process.start()

    def _update(self):
        while True:
            try:
                res = self.queue.get()
                if res is None:break
                update_iters, snapshot = res
                self.bar.update(update_iters, current_iter=update_iters, snapshot=utils.dict_to_params_str(snapshot, auto=False, sep=", "))
            except BaseException as e:
                self.bar.finish()
                if not isinstance(e, KeyboardInterrupt):
                    traceback.print_exc()
                sys.exit(1)

    def update(self, update_iters:int, snapshot:dict):
        self.queue.put((update_iters, snapshot))

    def finish(self):
        self.queue.put(None)
        self.bar.finish()