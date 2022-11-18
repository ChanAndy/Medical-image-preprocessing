#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Python 
@File    ：Decision_curve_multi.py
@IDE     ：PyCharm 
@Author  ：Andy
@Date    ：2022/11/15 15:11 
@Maibox  : 13266082905@163.com
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def decision_curve_analysis(y_pred_countinuous, y_test, p_min, p_max, epsilon):
    """
    Calculate the Net Benefit for a binary classifier
    :param clf: Binary classifier (scikit-learn)
    :param X_test: Independent features (Test set)
    :param y_test: Target vector (Test set)
    :param p_min: Lower limit for the threshold probability
    :param p_max: Upper limit for the threshold probability
    :param epsilon: Increase in the threshold probability for calculating the net benefit
    :return: Values for the threshold probabilities and their corresponding net benefit
    """

    p_serie = []
    net_benefit_serie = []
    for p in np.arange(p_min, p_max, epsilon):
        # y_pred = clf.predict_proba(X_test)[:, 1] > p
        y_pred = y_pred_countinuous > p
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        n = tn + fp + fn + tp
        net_benefit = (tp / n) - (fp / n) * (p / (1 - p))
        p_serie.append(p)
        net_benefit_serie.append(net_benefit)

    return p_serie, net_benefit_serie

# Net Benefit Prioritise All Referrals
def calculate_net_benefit_all(tp_test, tn_test, p_min, p_max, epsilon):
    """
    Calculate the Net Benefit for the 'Treat-All' clinical alternative
    :param tp_test: Number of True Positives in the Test set
    :param tn_test: Number of True Negative in the Test set
    :param p_min: Lower limit for the threshold probability
    :param p_max: Upper limit for the threshold probability
    :param epsilon: Increase in the threshold probability for calculating the net benefit
    :return: Values for the threshold probabilities and their corresponding net benefit
    """

    p_serie = []
    net_benefit_serie = []
    total = tp_test + tn_test
    for p in np.arange(p_min, p_max, epsilon):
        net_benefit = (tp_test / total) - (tn_test / total) * (p / (1 - p))
        p_serie.append(p)
        net_benefit_serie.append(net_benefit)

    return p_serie, net_benefit_serie


def plot_decision_curves(y_pred_countinuous, y_test, legend, p_min, p_max, epsilon, net_benefit_lower, net_benefit_upper, savefig, savefig_path):
    """
    Plotting the Net Benefit for a List of Classifiers (scikit-learn)
    :param clfs: List of binary classifiers
    :param labels: List of names (for including in the graph legend)
    :param X_test: Independent features (Test set)
    :param y_test: Target vector (Test set)
    :param p_min: Lower limit for the threshold probability
    :param p_max: Upper limit for the threshold probability
    :param epsilon: Increase in the threshold probability for calculating the net benefit
    :param net_benefit_lower: Lower limit for the Net Benefit (y axis)
    :param net_benefit_upper: Upper limit for the Net Benefit (y axis)
    :return: Decision Curve Analysis Plot
    """
    # Calculating True Positives and True Negatives (Test Set)
    tp_test = np.sum(y_test)
    tn_test = y_test.shape[0] - tp_test

    # Defining Figure Size
    plt.figure(figsize=(15, 10), dpi=80)

    # color = ["red", "blue", "yellow"]
    # Plotting each Classifier
    for i in range(0, len(legend)):
        print("legend:{}".format(legend[i]))
        p, net_benefit = decision_curve_analysis(y_pred_countinuous, y_test, p_min, p_max, epsilon)
        plt.plot(p, net_benefit, label=legend[i], color='red', linewidth=3)

    # Plotting Prioritise None
    plt.hlines(y=0, xmin=p_min, xmax=p_max, label='Treat None', linestyles='--', color='black', linewidth=2)

    # Plotting Prioritise All
    p_all, net_benefit_all = calculate_net_benefit_all(tp_test, tn_test, p_min, p_max, epsilon)
    plt.plot(p_all, net_benefit_all, label='Treat All', linestyle='dashed', color='black', linewidth=2)

    # Figure Configuration
    plt.xlabel('Threshold Probability', fontdict={'size': 30})
    plt.ylabel('Net Benefit', fontdict={'size': 30})
    # plt.title('Decision Curve Analysis')
    plt.ylim([net_benefit_lower, net_benefit_upper])
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)
    axes = plt.gca()
    plt.legend(fontsize=28)
    # 是否保存图像
    if savefig:
        plt.savefig(savefig_path)
    return plt.show()


# load the calculated data
df = pd.read_excel(r".\xxx.xlsx", sheet_name='Sheet1')
Read_1_train = df['average_Pre']     # continuous variable
pred_train = df['pred__nor']   # continuous variable
y_label = df['label']    # only label 0 and 1

# parameter initialization
p_min = 0
p_max = 1
epsilon = 0.01
net_benefit_lower = -0.1
net_benefit_upper = 0.65

# Count the number of labels 0 and 1, respectively
tp_test = np.sum(y_label)
tn_test = y_label.shape[0] - tp_test

plt.figure(figsize=(15, 10), dpi=80)

# DL model plt
p, net_benefit = decision_curve_analysis(pred_train, y_label, p_min, p_max, epsilon)
plt.plot(p, net_benefit, label="DL model", color='red', linewidth=3)
# othor model plt
p, net_benefit = decision_curve_analysis(Read_1_train, y_label, p_min, p_max, epsilon)
plt.plot(p, net_benefit, label="other model", color='green', linewidth=3)

# Plotting Prioritise None
plt.hlines(y=0, xmin=p_min, xmax=p_max, label='Treat None', linestyles='--', color='black', linewidth=2)

# Plotting Prioritise All
p_all, net_benefit_all = calculate_net_benefit_all(tp_test, tn_test, p_min, p_max, epsilon)
plt.plot(p_all, net_benefit_all, label='Treat All', linestyle='dashed', color='black', linewidth=2)

# Figure Configuration
plt.xlabel('Threshold Probability', fontdict={'size': 30})
plt.ylabel('Net Benefit', fontdict={'size': 30})
plt.ylim([net_benefit_lower, net_benefit_upper])
plt.tick_params(axis='x', labelsize=24)
plt.tick_params(axis='y', labelsize=24)
axes = plt.gca()
plt.legend(fontsize=28)

# whether save the fig
savefig = False
savefig_path = r"D:\xx.tif"
if savefig:
    plt.savefig(savefig_path)
plt.show()

