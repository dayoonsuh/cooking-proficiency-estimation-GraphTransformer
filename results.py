#! /usr/bin/python3
# Author : Kevin Feghoul

import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import os
from typing import *
from sklearn.metrics import confusion_matrix

from utils import *




def save_results(history:dict, id:int, ground_truth:np.ndarray, preds:np.ndarray, results_path:str) -> np.array:
    
    cm = confusion_matrix(ground_truth, preds)

    results_visualization = os.path.join(results_path, 'plots')

    make_dirs(results_visualization)
  
    plt.plot(history['loss'], label="Train loss")
    plt.plot(history['val_loss'], label="Test loss")
    leg = plt.legend(loc="best")
    plt.xlabel("Num epochs")
    plt.ylabel("Loss")
    plt.title("Loss over the number of epoch")
    plt.savefig(os.path.join(results_visualization, str(id) + '_loss'))
    plt.close()

    plt.plot(history['acc'], label="Train accuracy")
    plt.plot(history['val_acc'], label="Test accuracy")
    leg = plt.legend(loc="best")
    plt.xlabel("Num epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over the number of epoch")
    plt.savefig(os.path.join(results_visualization, str(id) + '_accuracy'))
    plt.close()

    plt.plot(history['F1'], label="F1 score")
    plt.plot(history['val_F1'], label="F1 score")
    leg = plt.legend(loc="best")
    plt.xlabel("Num epochs")
    plt.ylabel("F1 score")
    plt.title("F1 score over the number of epoch")
    plt.savefig(os.path.join(results_visualization, str(id) + '_F1'))
    plt.close()

    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title("Confusion matrix best accuracy")
    plt.savefig(os.path.join(results_visualization, str(id) + '_confusion_matrix'))
    plt.close()

    return cm
    

def results_all(all_acc:List[float], all_f1:List[float], all_precision:List[float], all_recall:List[float], model_name:str, save_results_path:str):

    avg_acc, avg_f1, avg_precision, avg_recall = round(np.mean(all_acc), ndigits=2), round(np.mean(all_f1), ndigits=2), round(np.mean(all_precision), ndigits=2), round(np.mean(all_recall), ndigits=2)
    std_acc, std_f1, std_precision, std_recall  = round(np.std(all_acc), ndigits=2), 100 *round(np.std(all_f1), ndigits=2), round(np.std(all_precision), ndigits=2), round(np.std(all_recall), ndigits=2)

    print('\n\nAverage accuracy : {}, Std accuracy : {}'.format(avg_acc, std_acc))
    print('Average F1 : {}, Std F1 : {}'.format(avg_f1, std_f1))
    print('Average precision : {}, Std precision : {}'.format(avg_precision, std_precision))
    print('Average recall : {}, Std recall : {}'.format(avg_recall, std_recall))

    results = [avg_acc, avg_f1, avg_precision]

    barplot = sns.barplot(x=['Average_acc', 'Average_F1'], y=results)
    for xtick in barplot.get_xticks():
        barplot.text(xtick, results[xtick], results[xtick], horizontalalignment='center',size='small',color='black',weight='semibold')
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_results_path, model_name + '_scores'))
    plt.close()



def save_results_all_folds(all_results:list[float], model_name:str, save_results_path:str, all_results_path:str, best_results_path:str, window_size:int, overlap:int) -> None:

    
    all_acc, all_f1, all_prec, all_rec, all_cm = all_results
    
    avg_acc, avg_f1, avg_prec, avg_rec = round(np.mean(all_acc), ndigits=2), round(np.mean(all_f1), ndigits=2), round(np.mean(all_prec), ndigits=2), round(np.mean(all_rec), ndigits=2)
    std_acc, std_f1, std_prec, std_rec  = round(np.std(all_acc), ndigits=2), round(np.std(all_f1), ndigits=2), round(np.std(all_prec), ndigits=2), round(np.std(all_rec), ndigits=2)

    print('\n\nAverage F1 : {}, Std F1 : {}'.format(avg_f1, std_f1))
    print('Average accuracy : {}, Std accuracy : {}'.format(avg_acc, std_acc))
    print('Average precision : {}, Std precision : {}'.format(avg_prec, std_prec))
    print('Average recall : {}, Std recall : {}'.format(avg_rec, std_rec))

    scores = [avg_f1, avg_acc, avg_prec, avg_rec, model_name, window_size, overlap, save_results_path]
    append_new_line(all_results_path, ', '.join('{}'.format(k) for k in scores))
    
    data = pd.read_csv(all_results_path, sep=",", header=None)
    data.columns = ['f1', 'acc', 'precision', 'recall', 'model_name', 'window_size', 'overlap', 'path']
    data.sort_values('acc', ascending=False, inplace=True)
    data.to_csv(best_results_path, index=None, sep=',')

    x_acc = ['fold_' + str(i) for i in range(1, len(all_acc)+1)] + ['avg_acc']
    x_f1 = ['fold_' + str(i) for i in range(1, len(all_f1)+1)] + ['avg_f1']
    x_precision = ['fold_' + str(i) for i in range(1, len(all_prec)+1)] + ['avg_precision']
    x_recall = ['fold_' + str(i) for i in range(1, len(all_rec)+1)] + ['avg_recalls']

    results_acc = all_acc + [avg_acc]
    results_f1 = all_f1 + [avg_f1]
    results_prec = all_prec + [avg_prec]
    results_rec = all_rec + [avg_rec]

    barplot = sns.barplot(x=x_acc, y=results_acc)
    for xtick in barplot.get_xticks():
        barplot.text(xtick, results_acc[xtick], results_acc[xtick], horizontalalignment='center',size='small',color='black',weight='semibold')
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_results_path, model_name + '_accuracy'))
    plt.close()

    barplot = sns.barplot(x=x_f1, y=results_f1)
    for xtick in barplot.get_xticks():
        barplot.text(xtick, results_f1[xtick], results_f1[xtick], horizontalalignment='center',size='small',color='black',weight='semibold')
    plt.ylabel("F1")
    plt.savefig(os.path.join(save_results_path, model_name + '_f1'))
    plt.close()

    barplot = sns.barplot(x=x_precision, y=results_prec)
    for xtick in barplot.get_xticks():
        barplot.text(xtick, results_prec[xtick], results_prec[xtick], horizontalalignment='center',size='small',color='black',weight='semibold')
    plt.ylabel("Precision")
    plt.savefig(os.path.join(save_results_path, model_name + '_precision'))
    plt.close()


    barplot = sns.barplot(x=x_recall, y=results_rec)
    for xtick in barplot.get_xticks():
        barplot.text(xtick, results_rec[xtick], results_rec[xtick], horizontalalignment='center',size='small',color='black',weight='semibold')
    plt.ylabel("Recall")
    plt.savefig(os.path.join(save_results_path, model_name + '_recall'))
    plt.close()

    sum_cm = np.zeros(all_cm[0].shape)
    for cm in all_cm:
        sum_cm += cm

    plt.figure(figsize=(7,5))
    sns.heatmap(sum_cm.astype('int64'), annot=True, fmt='d', cbar=False, cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title("Sum confusion matrix")
    plt.savefig(os.path.join(save_results_path, 'sum_confusion_matrix'))
    plt.close()







