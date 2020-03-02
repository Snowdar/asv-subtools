# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-11-25)

import os
import matplotlib.pyplot as plt


def plot_lr_finder(data, savefile:str=None, acc:bool=False, savedpi:int=256):
    """Used for learn rate finder.
       loss <=> lr

       data:
            [(log_lrs,losses,{"label":"baseline",...}), 
             (log_lrs,losses,{}),
             ...,
             (log_lrs,losses,{})]
    """
    fig, ax = plt.subplots()
    ax.set_title("LR Finder")

    for log_lrs, losses, param_dict in data:
        ax.plot(log_lrs, losses, **param_dict) # Set line formatter, color, label etc.

    ax.legend()  # Make legend valid
    ax.set_xlabel('Learning Rate')

    if acc:
        ax.set_ylabel('Accuracy')
    else:
        ax.set_ylabel('Loss')

    # Axis locator
    # Axis formatter
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('$10^{%d}$'))

    if savefile is not None:
        plt.savefig(savefile, dpi=savedpi)

    plt.show()
    


# def plot_training(log_csv:str, savefile:str=None, savedpi:int=256):
#     """Used for training loss/acc/lr.
#     """
#     # Data
#     df = utils.read_log_csv(log_csv)
#     df["epoch"].values + df["iter"].values

#     data_dict = {}

#     for head in df:
#         if head != "epoch" and head != "iter":
#             data_dict[head] = 


#     fig, ax = plt.subplots()
#     ax.set_title("Loss")
    

