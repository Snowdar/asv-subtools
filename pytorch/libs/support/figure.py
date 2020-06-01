# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-11-25)

import os
import matplotlib.pyplot as plt

# To do

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
    

# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# import pandas as pd
# # To do
# # xmuspeech Rui Nangong
#
# class PlotTrainingCSV():
#     def __init__(self, csv_path, is_cmp=False):
#         self.df = pd.read_csv(csv_path)
#         self.epoch_num = len(self.df[self.df['iter']==1])
#         self.iter_num = len(self.df[self.df['epoch']==1])
#         self.train_epoch = self.df['epoch']
#         self.train_iter = self.df['iter']
#         self.train_loss = list(self.df['train_loss'])
#         self.train_acc = list(self.df['train_acc'])
#         self.valid_loss_epoch = self.df[pd.isna(self.df['valid_loss']) == False]['epoch']
#         self.valid_loss_iter = self.df[pd.isna(self.df['valid_loss']) == False]['iter']
#         self.valid_loss = list(self.df[pd.isna(self.df['valid_loss']) == False]['valid_loss'])
#         self.valid_acc_epoch = self.df[pd.isna(self.df['valid_acc']) == False]['epoch']
#         self.valid_acc_iter = self.df[pd.isna(self.df['valid_acc']) == False]['iter']
#         self.valid_acc = list(self.df[pd.isna(self.df['valid_acc']) == False]['valid_acc'])
#         if is_cmp==False:
#             self.mainfig = plt.figure(figsize=(18,9))
#         else:
#             self.mainfig = None
#         self.cmp_DL = None

#     def smooth_loss(self, beta=0.9):
#         proto_train_loss = self.train_loss
#         smooth_train_loss = []
#         prosum = 0.0
#         for i in range(0, len(proto_train_loss)):
#             if(i == 0):
#                 smooth_train_loss.append((1 - beta) * proto_train_loss[0] / (1 - beta))
#                 prosum = smooth_train_loss[0]
#                 continue
#             prosum = prosum * beta + (1 - beta) * proto_train_loss[i]
#             smooth_train_loss.append(prosum)
#         self.train_loss = smooth_train_loss
#         self.df['train_loss'] = smooth_train_loss

#     def smooth_acc(self, beta=0.9):
#         proto_train_acc= self.train_acc
#         smooth_train_acc = []
#         prosum = 0.0
#         for i in range(0, len(proto_train_acc)):
#             if(i == 0):
#                 smooth_train_acc.append((1 - beta) * proto_train_acc[0] / (1 - beta))
#                 prosum = smooth_train_acc[0]
#                 continue
#             prosum = prosum * beta + (1 - beta) * proto_train_acc[i]
#             smooth_train_acc.append(prosum)
#         self.train_acc = smooth_train_acc
#         self.df['train_acc'] = smooth_train_acc

#     def draw_loss(self, row=1, col=1, no=1, x_index=4, x_maj=10000, x_min=5000, is_cmp=False):
#         if is_cmp==False:
#             fig = self.mainfig.add_subplot(row, col, no)

#         x_train_loss = (self.train_epoch - 1) * self.iter_num + self.train_iter
#         l_train_loss, = plt.plot(x_train_loss, self.train_loss, color='red', label='train_loss')
#         x_valid_loss = (self.valid_loss_epoch - 1) * self.iter_num + self.valid_loss_iter
#         l_valid_loss, = plt.plot(x_valid_loss, self.valid_loss, color='green',label='valid_loss')

#         grouped = self.df.groupby('epoch')
#         x_train_last = []
#         y_train_last = []
#         x_valid_last = []
#         y_valid_last = []
#         for e in range(1, self.epoch_num + 1):
#             epoch_group = grouped.get_group(e)
#             x_train_last.append((e - 1) * self.iter_num + int(epoch_group[-1:]['iter']))
#             y_train_last.append(float(epoch_group[-1:]['train_loss']))
#             x_valid_last.append((e - 1) * self.iter_num + int(epoch_group[pd.isna(epoch_group['valid_loss']) == False][-1:]['iter']))
#             y_valid_last.append(float(epoch_group[pd.isna(epoch_group['valid_loss']) == False][-1:]['valid_loss']))
#         plt.plot(pd.core.series.Series(x_train_last), pd.core.series.Series(y_train_last), 'x', label='train_loss (epoch)')
#         plt.plot(pd.core.series.Series(x_valid_last), pd.core.series.Series(y_valid_last), 'x', label='valid_loss (epoch)')

#         if is_cmp==False:
#             x_size = 10 ** x_index
#             def formatnum(x, pos):
#                 return '$%.1f$x$10^{%d}$' % (x/x_size, x_index)
#             formatter = plt.FuncFormatter(formatnum)
#             fig.xaxis.set_major_formatter(formatter)
#             xminorLocator = MultipleLocator(x_min)
#             xmajorLocator = MultipleLocator(x_maj)
#             fig.xaxis.set_minor_locator(xminorLocator)
#             fig.xaxis.set_major_locator(xmajorLocator)

#         plt.xlabel('iter')
#         plt.ylabel('loss')
#         plt.title('Loss Curve')
#         plt.grid()
#         plt.legend()

#     def draw_acc(self, row=1, col=1, no=1, x_index=4, x_maj=10000, x_min=5000, is_cmp=False):
#         if is_cmp==False:
#             fig = self.mainfig.add_subplot(row, col, no)

#         x_train_acc = (self.train_epoch - 1) * self.iter_num + self.train_iter
#         l_train_acc, = plt.plot(x_train_acc, self.train_acc, color='red', label='train_acc')
#         x_valid_acc = (self.valid_acc_epoch - 1) * self.iter_num + self.valid_acc_iter
#         l_valid_acc, = plt.plot(x_valid_acc, self.valid_acc, color='green', label='valid_acc')

#         grouped = self.df.groupby('epoch')
#         x_train_last = []
#         y_train_last = []
#         x_valid_last = []
#         y_valid_last = []
#         for e in range(1, self.epoch_num + 1):
#             epoch_group = grouped.get_group(e)
#             x_train_last.append((e - 1) * self.iter_num + int(epoch_group[-1:]['iter']))
#             y_train_last.append(float(epoch_group[-1:]['train_acc']))
#             x_valid_last.append((e - 1) * self.iter_num + int(epoch_group[pd.isna(epoch_group['valid_acc']) == False][-1:]['iter']))
#             y_valid_last.append(float(epoch_group[pd.isna(epoch_group['valid_acc']) == False][-1:]['valid_acc']))
#         plt.plot(pd.core.series.Series(x_train_last), pd.core.series.Series(y_train_last), 'x', label='train_acc (epoch)')
#         plt.plot(pd.core.series.Series(x_valid_last), pd.core.series.Series(y_valid_last), 'x', label='valid_acc (epoch)')

#         if is_cmp==False:
#             x_size = 10 ** x_index
#             def formatnum(x, pos):
#                 return '$%.1f$x$10^{%d}$' % (x/x_size, x_index)
#             formatter = plt.FuncFormatter(formatnum)
#             fig.xaxis.set_major_formatter(formatter)
#             xminorLocator = MultipleLocator(x_min)
#             xmajorLocator = MultipleLocator(x_maj)
#             fig.xaxis.set_minor_locator(xminorLocator)
#             fig.xaxis.set_major_locator(xmajorLocator)

#         plt.xlabel('iter')
#         plt.ylabel('accuracy (%)')
#         plt.title('Inverted Specaugment (f=0.2, t=0.2, random multi=4,4)')
#         plt.grid()
#         plt.ylim(60,101)
#         plt.legend(loc=4)
    
#     def my_show(self):
#         plt.show()

#     def draw_all(self, x_index=4, x_maj=10000, x_min=5000):
#         self.draw_loss(row=1, col=2, no=1, x_index=x_index, x_maj=x_maj, x_min=x_min)
#         self.draw_acc(row=1, col=2, no=2, x_index=x_index, x_maj=x_maj, x_min=x_min)
#         self.my_show()

#     def draw_cmp(self, cmp_path, task='loss', x_index=4, x_maj=10000, x_min=5000):
#         self.cmp_DL = Draw_csv(cmp_path, is_cmp=True)
#         if task=='loss':
#             self.draw_loss(row=1, col=2, no=1)
#             fig = self.mainfig.add_subplot(1,2,2)
#             self.cmp_DL.draw_loss(is_cmp=True)
#             x_size = 10 ** x_index
#             def formatnum(x, pos):
#                 return '$%.1f$x$10^{%d}$' % (x/x_size, x_index)
#             formatter = plt.FuncFormatter(formatnum)
#             fig.xaxis.set_major_formatter(formatter)
#             xminorLocator = MultipleLocator(x_min)
#             xmajorLocator = MultipleLocator(x_maj)
#             fig.xaxis.set_minor_locator(xminorLocator)
#             fig.xaxis.set_major_locator(xmajorLocator)
#             self.my_show()
#         if task=='acc':
#             self.draw_acc(row=1, col=2, no=1)
#             fig = self.mainfig.add_subplot(1,2,2)
#             self.cmp_DL.draw_acc(is_cmp=True)
#             x_size = 10 ** x_index
#             def formatnum(x, pos):
#                 return '$%.1f$x$10^{%d}$' % (x/x_size, x_index)
#             formatter = plt.FuncFormatter(formatnum)
#             fig.xaxis.set_major_formatter(formatter)
#             xminorLocator = MultipleLocator(x_min)
#             xmajorLocator = MultipleLocator(x_maj)
#             fig.xaxis.set_minor_locator(xminorLocator)
#             fig.xaxis.set_major_locator(xmajorLocator)
#             self.my_show()
#         if task=='loss and acc':
#             self.draw_loss(row=2, col=2, no=1)
#             fig = self.mainfig.add_subplot(2,2,3)
#             self.cmp_DL.draw_loss(is_cmp=True)
#             x_size = 10 ** x_index
#             def formatnum(x, pos):
#                 return '$%.1f$x$10^{%d}$' % (x/x_size, x_index)
#             formatter = plt.FuncFormatter(formatnum)
#             fig.xaxis.set_major_formatter(formatter)
#             xminorLocator = MultipleLocator(x_min)
#             xmajorLocator = MultipleLocator(x_maj)
#             fig.xaxis.set_minor_locator(xminorLocator)
#             fig.xaxis.set_major_locator(xmajorLocator)
#             self.draw_acc(row=2, col=2, no=2)
#             fig = self.mainfig.add_subplot(2,2,4)
#             self.cmp_DL.draw_acc(is_cmp=True)
#             x_size = 10 ** x_index
#             def formatnum(x, pos):
#                 return '$%.1f$x$10^{%d}$' % (x/x_size, x_index)
#             formatter = plt.FuncFormatter(formatnum)
#             fig.xaxis.set_major_formatter(formatter)
#             xminorLocator = MultipleLocator(x_min)
#             xmajorLocator = MultipleLocator(x_maj)
#             fig.xaxis.set_minor_locator(xminorLocator)
#             fig.xaxis.set_major_locator(xmajorLocator)
#             self.my_show()