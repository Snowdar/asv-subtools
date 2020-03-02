# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2020-02-20)

import math
import logging
import numpy as np
import progressbar

from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from .reporter import LRFinderReporter

# Wrap stderr before logger init.
progressbar.streams.wrap_stderr()

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def for_lr_finder(function):
    """A decorator to use lr finder conveniently.
    Self means using this decorator in trainer class.
    Reference: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html.
    """
    def wrapper(self, trn_loader, optimizer, init_lr=1e-6, final_lr=10., num_iters=None, beta=0.98, split=[5,-10]):
        if init_lr < 0:
            raise ValueError("Expected init_lr > 0, but got init_lr = {}.".format(init_lr))
        if final_lr < init_lr:
            raise ValueError("Expected final_lr > init_lr, but got final_lr {} <= and init_lr {}.".format(final_lr, init_lr))
        if num_iters is not None and num_iters <= 1:
            raise ValueError("Expected num_iters > 1, but got {}.".format(num_iters))
        if not isinstance(trn_loader, DataLoader):
            raise TypeError("Expected Dataloader, but got {}.".format(type(trn_loader).__name__))
        if not isinstance(optimizer, Optimizer):
            raise TypeError("Expected Optimizer, but got {}.".format(type(Optimizer).__name__))

        # If num_iters is None, then just run one epoch.
        if num_iters is not None:
            num_iters = num_iters
            epochs = (num_iters - 1) // len(trn_loader) + 1
        else:
            num_iters = len(trn_loader)
            epochs = 1

        logger.info("Run lr finder from init_lr = {} to final_lr = {} with {} iters.".format(init_lr, final_lr, num_iters))

        # Init.
        mult = (final_lr / init_lr) ** (1 / (num_iters - 1))

        num_batch = 0
        avg_values = 0.
        log_lrs = []

        reporter = LRFinderReporter(num_iters)

        # Start.
        lr = init_lr
        optimizer.param_groups[0]['lr'] = lr

        for this_epoch in range(epochs):
            for batch in trn_loader:
                num_batch += 1

                # The values is a vector of numpy and function return a list of float values.
                values = np.array(function(self, batch))

                # Compute the smoothed values. The avg_values will be also a vector of numpy rather than 0.
                avg_values = beta * avg_values + (1 - beta) * values
                smoothed_values = avg_values / (1 - beta ** num_batch)

                snapshot = {"lr":lr, "metric":smoothed_values[0]}
                reporter.update(num_batch, snapshot)

                # Stop if the main value is exploding.
                if num_batch > 1 and smoothed_values[0] > 4 * best_value:
                    reporter.finish()
                    logger.info("Stop lr finder early by default rule.")
                    return log_lrs[split[0]:split[1]], value_matrix.T[:,split[0]:split[1]]

                # Record the best main value. The main value which has the index-0 is usually the training loss.
                if num_batch == 1 or smoothed_values[0] < best_value:
                    best_value = smoothed_values[0]

                # Store the values.
                if num_batch == 1:
                    value_matrix = smoothed_values
                else:
                    value_matrix = np.vstack([value_matrix, smoothed_values])

                log_lrs.append(math.log10(lr))

                if num_batch >= num_iters:
                    reporter.finish()
                    return log_lrs[split[0]:split[1]], value_matrix.T[:,split[0]:split[1]]

                # Update the lr for the next step.
                lr *= mult
                optimizer.param_groups[0]['lr'] = lr

        reporter.finish()
        return log_lrs[split[0]:split[1]], value_matrix.T[:,split[0]:split[1]]
    return wrapper