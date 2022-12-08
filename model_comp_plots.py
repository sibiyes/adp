import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

plt.rcParams.update({'font.size': 22})

### generate comparison plots (precision-recall and ROC curve) for a set of models
def main():
    print(5)

    # model_tags = ['log_reg', 'gb', 'rf_num_est=100_md=5']
    # tag = 'model_basic_comp'

    # model_tags = ['log_reg', 'log_reg-log']
    # tag = 'model_lr_log_comp'

    # model_tags = ['rf_num_est=50_md=5', 'rf_num_est=75_md=5', 'rf_num_est=100_md=5']
    # tag = 'model_rf_param_comp'

    model_tags = ['log_reg', 'log_reg-full', 'gb', 'gb-full']
    tag = 'model_cv_comp'

    plots_folder = base_folder + '/plots/model_comp'

    metrics_all = []
    roc_all = []
    for model_tag in model_tags:
        model_output_folder = base_folder + '/model_output/{0}'.format(model_tag)

        metrics = pd.read_csv(model_output_folder + '/metrics.csv'.format())
        roc = pd.read_csv(model_output_folder + '/roc.csv'.format())

        metrics['model_tag'] = model_tag
        roc['model_tag'] = model_tag

        metrics_all.append(metrics)
        roc_all.append(roc)

    
    metrics_all = pd.concat(metrics_all)
    roc_all = pd.concat(roc_all)

    print(metrics_all)
    print(roc_all)

    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    ax = ax.ravel()

    sns.lineplot(data = metrics_all, x = 'prob_threshold', y = 'precision', hue = 'model_tag', ax  = ax[0])
    sns.scatterplot(data = metrics_all, x = 'prob_threshold', y = 'precision', hue = 'model_tag', legend = False, ax  = ax[0])
    sns.lineplot(data = metrics_all, x = 'prob_threshold', y = 'recall', hue = 'model_tag', ax  = ax[1])
    sns.scatterplot(data = metrics_all, x = 'prob_threshold', y = 'recall', hue = 'model_tag', legend = False, ax  = ax[1])

    #plt.show()

    plt.gcf().set_size_inches(20, 10)
    plt.savefig(plots_folder + '/precision_recall_{0}.jpg'.format(tag))
    plt.clf()

    ax = sns.lineplot(data = roc_all, x = 'fpr', y = 'tpr', hue = 'model_tag')
    ax.set_title('ROC Curve')
    
    #plt.show()

    plt.gcf().set_size_inches(20, 10)
    plt.savefig(plots_folder + '/roc_curve_{0}.jpg'.format(tag))
    plt.clf()
    

if __name__ == '__main__':
    main()