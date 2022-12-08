import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

### Compute model prediction metrics for evaluation
### computes accuracy, precision and recall
def pred_results(y_true, y_pred, categories):
    conf_matrix = confusion_matrix(y_true, y_pred, labels = categories)
    print(conf_matrix)
    correct_pred = np.trace(conf_matrix)
    incorrect_pred = np.sum(conf_matrix) - np.trace(conf_matrix)
    print(correct_pred/np.sum(conf_matrix), incorrect_pred/np.sum(conf_matrix), np.sum(conf_matrix))

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)

    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)

    return accuracy, precision, recall

### Compute precision, recall and roc values and generate plots for the same
### precision and recall values are generated for multiple prediction probability threshold
def run_evaluation(pred_probs, target, model_tag):
    pred_thresh = np.sum(target)/len(target)
    preds = np.array([1.0 if p > pred_thresh else 0.0 for p in pred_probs])
    pred_results(target, preds, [0.0, 1.0])

    thresh_vals = list(np.arange(0.025, 0.05, 0.0025)) + list(np.arange(0.06, 0.1, 0.01)) + list(np.arange(0.15, 0.5, 0.05))
    print(thresh_vals)

    accuracy_vals = []
    precision_vals = []
    recall_vals = []
    for p_thresh in thresh_vals:
        print('Prediction using thresh = {0}'.format(np.round(p_thresh, 4)))
        preds = np.array([1.0 if p > p_thresh else 0.0 for p in pred_probs])
        accuracy, precision, recall = pred_results(target, preds, [0.0, 1.0])

        accuracy_vals.append(accuracy)
        precision_vals.append(precision)
        recall_vals.append(recall)

    plots_folder = base_folder + '/plots/model'
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    ax = ax.ravel()

    metrics_df = pd.DataFrame({'prob_threshold': thresh_vals, 'precision': precision_vals, 'recall': recall_vals})

    sns.lineplot(data = metrics_df, x = 'prob_threshold', y = 'precision', ax  = ax[0])
    sns.scatterplot(data = metrics_df, x = 'prob_threshold', y = 'precision', ax  = ax[0])
    sns.lineplot(data = metrics_df, x = 'prob_threshold', y = 'recall', ax  = ax[1])
    sns.scatterplot(data = metrics_df, x = 'prob_threshold', y = 'recall', ax  = ax[1])

    fig.suptitle('Precision and Recall by Prediction Probability Threshold')

    #plt.show()

    plt.gcf().set_size_inches(20, 10)
    plt.savefig(plots_folder + '/precision_recall_{0}.jpg'.format(model_tag))
    plt.clf()

    fpr, tpr, thresholds = metrics.roc_curve(target, pred_probs)
    roc_auc = metrics.roc_auc_score(target, pred_probs)
    print('ROC AUC:', roc_auc)
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})

    ax = sns.lineplot(x = fpr, y = tpr)
    ax.set_title('ROC Curve - AUC: {0}'.format(np.round(roc_auc, 4)))
    #plt.show()

    plt.gcf().set_size_inches(20, 10)
    plt.savefig(plots_folder + '/roc_curve_{0}.jpg'.format(model_tag))
    plt.clf()

    output_folder = base_folder + '/model_output/{0}'.format(model_tag)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    metrics_df.to_csv(output_folder + '/metrics.csv', index = None)
    roc_df.to_csv(output_folder + '/roc.csv', index = None)


### Train a model on the full dataset and generate model evaluation metrics
def train_model_full(features, target, model_type, params, model_tag):
    if (model_type == 'log_reg'):
        model = LogisticRegression(max_iter = 1000, solver = 'liblinear', C = 2.0)
    elif (model_type == 'gb'):
        model = GradientBoostingClassifier(n_estimators = 100, max_depth = 5)
    elif (model_type == 'rf'):
        if (params is None):
            model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state=0)
        else:
            model = RandomForestClassifier(n_estimators = params['num_est'], max_depth = params['md'], random_state=0)
    else:
        print('Unsupported Model Type')
        sys.exit(0)

    model.fit(features, target)
    pred_probs = model.predict_proba(features)[:, 1]

    run_evaluation(pred_probs, target, model_tag)

    return model

### print and store feature coefficients of feature importance scores based on the model type
def analyze_model_features(model, feature_cols, model_type):
    output_folder = base_folder + '/model_features_output'
    if (model_type == 'log_reg'):
        feature_coef = pd.DataFrame({'features': feature_cols, 'coef': list(model.coef_[0])})
        print(feature_coef)

        feature_coef.to_csv(output_folder + '/log_reg_coef.csv', index = None)

    if (model_type in ['rf', 'gb']):
        feature_importance = pd.DataFrame({'features': feature_cols, 'importance_score': model.feature_importances_})
        feature_importance = feature_importance.sort_values(by = ['importance_score'], ascending = False)
        print(feature_importance)

        feature_importance.to_csv(output_folder + '/feature_importance_{0}.csv'.format(model_type), index = None)

### Run k-fold cross validation on the dataset based on the specified 
### modeling approach. Build models on cross validated dataset and
### compute model evaluation metrics
def run_model_cv(features, target, model_type, params, model_tag):
    k = 5
    n = features.shape[0]
    cv_indices = list(np.arange(k))*(n//k + 1)
    cv_indices = np.array([int(i) for i in cv_indices[:n]])

    print(cv_indices)
    print(len(cv_indices))

    true_target_cv = []
    pred_probs_cv = []

    for cv_ind in range(k):
        print('Cross Validation:', cv_ind)
        
        features_train = np.copy(features[cv_indices != cv_ind])
        features_test = np.copy(features[cv_indices == cv_ind])

        target_train = np.copy(target[cv_indices != cv_ind])
        target_test = np.copy(target[cv_indices == cv_ind])

        if (model_type == 'log_reg'):
            model = LogisticRegression(max_iter = 1000, solver = 'liblinear', C = 2.0)
        elif (model_type == 'gb'):
            model = GradientBoostingClassifier(n_estimators = 100, max_depth = 5)
        elif (model_type == 'rf'):
            if (params is None):
                model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state=0)
            else:
                model = RandomForestClassifier(n_estimators = params['num_est'], max_depth = params['md'], random_state=0)
        else:
            print('Unsupported Model Type')
            sys.exit(0)

        model.fit(features_train, target_train)
        pred_probs = model.predict_proba(features_test)[:, 1]

        true_target_cv.append(target_test)
        pred_probs_cv.append(pred_probs)

    pred_probs = np.hstack(pred_probs_cv)
    true_target = np.hstack(true_target_cv)

    run_evaluation(pred_probs, true_target, model_tag)


def main():
    data_file = base_folder + '/CaseSTudy_2_data_no_dup.xlsx'
    data = pd.read_excel(data_file)
    data = data.drop(columns = ['Channel_Grouping'])

    ### parameters for model run

    # model_type = 'log_reg'
    # params = None

    model_type = 'rf'
    params = {'num_est': 100, 'md': 5}

    cv = True
    log_transform = False


    model_tag = model_type
    if (params is not None):
        params_tag = []
        for k, v in params.items():
            params_tag.append('{0}={1}'.format(k, v))

        model_tag = model_tag + '_' + '_'.join(params_tag)

    col_types = data.dtypes.to_dict()

    feature_cols = data.columns[1:-1]
    print(feature_cols)

    numeric_cols = [c for c in feature_cols if col_types[c] in ('float', 'int64')]
    categorical_cols = [c for c in feature_cols if col_types[c] == 'object']
    target_col = data.columns[-1]
    
    print(numeric_cols)
    print(categorical_cols)
    print(target_col)

    data_numeric = data[numeric_cols]
    data_categorical = data[categorical_cols]
    data_target = data[[target_col]]

    if (log_transform):
        model_tag += '-log'
        for col in numeric_cols:
            data_numeric[col + '_log'] = np.log(data[col] + 0.0001)

    print(data_categorical)

    enc = OneHotEncoder()
    enc.fit(data_categorical)

    data_categorical_oh = enc.transform(data_categorical).toarray()
    data_categorical_oh = pd.DataFrame(data_categorical_oh)

    cols_enc = []
    for col, levels in zip(data_categorical.columns, enc.categories_):
        print(col)
        print(levels)

        cols_enc += [col + '_' + l for l in levels]

    print(cols_enc)

    data_categorical_oh.columns = cols_enc

    print(print(data_categorical_oh))

    data_model = pd.concat((data_numeric, data_categorical_oh, data_target), axis = 1)
    print(data_model)
    
    feature_col_names = data_model.columns[:-1]

    features = data_model[feature_col_names].values
    target = data_model[data_model.columns[-1]].values

    if (cv == True):
        run_model_cv(features, target, model_type, params, model_tag)
    else:
        model = train_model_full(features, target, model_type, params, model_tag + '-full')
        analyze_model_features(model, feature_col_names, model_type)

    sys.exit(0)
         



if __name__ == '__main__':
    main()