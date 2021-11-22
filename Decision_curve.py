#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : Decision_curve.py
@Author: Andy
@Date  : 2021/11/22
@Desc  :
@Contact : 1369635303@qq.com
"""

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def decision_curve_analysis(y_pred_countinuous, y_label, p_min, p_max, epsilon):
    """
    Calculate the Net Benefit
    :param y_pred_countinuous: Categorical Predictive Value
    :param y_label: Target vector (Test set)
    :param p_min: Lower limit for the threshold probability
    :param p_max: Upper limit for the threshold probability
    :param epsilon: Increase in the threshold probability for calculating the net benefit
    :return: Values for the threshold probabilities and their corresponding net benefit
    """

    p_serie = []
    net_benefit_serie = []
    for p in np.arange(p_min, p_max, epsilon):
        y_pred = y_pred_countinuous > p
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred).ravel()
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


def plot_decision_curves(y_pred_countinuous, y_label, legend, p_min, p_max, epsilon, net_benefit_lower,
                         net_benefit_upper, savefig, savefig_path):
    """
    Plotting the Net Benefit

    :param y_pred_countinuous: Categorical Predictive Value
    :param y_label: Target vector (Test set)
    :param legend: Name of task
    :param p_min: Lower limit for the threshold probability
    :param p_max: Upper limit for the threshold probability
    :param epsilon: Increase in the threshold probability for calculating the net benefit
    :param net_benefit_lower: Lower limit for the Net Benefit (y axis)
    :param net_benefit_upper: Upper limit for the Net Benefit (y axis)
    :param savefig: Save Picture
    :param savefig_path: Path to Save Picture
    :return: Decision Curve Analysis Plot
    """
    # Calculating True Positives and True Negatives (Test Set)
    tp_test = np.sum(y_label)

    try:
        if isinstance(y_label, list):
            tn_test = len(y_label) - tp_test
        elif type(y_label) is np.ndarray:
            tn_test = y_label.shape[0] - tp_test
        # elif
        else:
            return TypeError
    except TypeError:
        print("If you want to enter data types other than list and array, modify the code here")

    # Defining Figure Size
    plt.figure(figsize=(15, 10), dpi=80)

    color = ["red"]
    # Plotting each Classifier
    for i in range(0, len(legend)):
        print("legend:{}".format(legend[i]))
        p, net_benefit = decision_curve_analysis(y_pred_countinuous, y_label, p_min, p_max, epsilon)
        plt.plot(p, net_benefit, label=legend[i], color=color[i], linewidth=3)

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
    plt.legend(fontsize=28)
    # 是否保存图像
    if savefig:
        plt.savefig(savefig_path)
    return plt.show()


if __name__ == "__main__":
    pred_score = [0.317484, 0.251464, 0.38303600000000004, 0.38526, 0.48041, 0.563246, 0.618552, 0.50634, 0.6341280000000001, 0.689528, 0.747782, 0.181108, 0.48924599999999996, 0.76515, 0.586132, 0.77895, 0.49177, 0.6268779999999999, 0.61701, 0.693028, 0.485454, 0.6608040000000001, 0.664062, 0.713176, 0.727976, 0.735674, 0.7466940000000001, 0.809594]
    y_label = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    tp_test = np.sum(y_label)
    tn_test = len(y_label) - tp_test

    """Calculate the Net Benefit"""
    p_train, net_train = decision_curve_analysis(pred_score, y_label, 0, 1, 0.01)

    """Plotting the Net Benefit"""
    p_train_all, net_train_all = calculate_net_benefit_all(tp_test, tn_test, 0, 1, 0.01)
    legend = ["DL model"]

    """Whether to save the image"""
    savefig = True
    savefig_path = "./Decision_curve.tif"
    Decision_curve_plt = plot_decision_curves(pred_score, y_label, legend, 0, 1, 0.01, -0.1, 0.8, savefig, savefig_path)
