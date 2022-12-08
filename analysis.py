import os
import sys
from copy import copy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

### Plot density and histogram of counts for variables in the dataset
def distribution_plots():
    data_file = base_folder + '/CaseSTudy_2_data_no_dup.xlsx'
    data = pd.read_excel(data_file)

    plots_folder = base_folder + '/plots'

    plt.rcParams.update({'font.size': 22})

    data['Visitor_Identifier'] = data['Visitor_Identifier'].astype(str)
    data['Lead _Form_submission_str'] = data['Lead _Form_submission'].astype(str)


    col_types = data.dtypes.to_dict()

    ### Plot the density of the log transformed session variables

    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    sns.kdeplot(data = data, x = 'Avg_Session_Duration', ax = ax[0][0])
    sns.kdeplot(data = data, x = 'avg_time_on_page', ax = ax[0][1])
    sns.kdeplot(data = data, x = 'Pages_Session', ax = ax[1][0])
    sns.kdeplot(data = data, x = 'pageviews', ax = ax[1][1])

    #plt.show()

    plt.gcf().set_size_inches(20, 10)
    plt.savefig(plots_folder + '/density_session.jpg')
    plt.clf()

    numeric_cols = [c for c in col_types if col_types[c] in ('float', 'int64')]
    print(numeric_cols)

    for col in numeric_cols:
        data[col + '_log'] = np.log(data[col] + 0.0001)

    print(data)

    ### Plot the density of the log transformed session variables

    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    sns.kdeplot(data = data, x = 'Avg_Session_Duration_log', ax = ax[0][0])
    sns.kdeplot(data = data, x = 'avg_time_on_page_log', ax = ax[0][1])
    sns.kdeplot(data = data, x = 'Pages_Session_log', ax = ax[1][0])
    sns.kdeplot(data = data, x = 'pageviews_log', ax = ax[1][1])

    #plt.show()

    plt.gcf().set_size_inches(20, 10)
    plt.savefig(plots_folder + '/density_session_log.jpg')
    plt.clf()


    ### plot the density of session plus variables

    fig, ax = plt.subplots(nrows = 2, ncols = 3)
    sns.kdeplot(data = data, x = 'Session_1plus_minute', ax = ax[0][0])
    sns.kdeplot(data = data, x = 'Session_3plus_minutes', ax = ax[0][1])
    sns.kdeplot(data = data, x = 'Session_3plus_pages', ax = ax[0][2])
    sns.kdeplot(data = data, x = 'Session_5plus_minutes', ax = ax[1][0])
    sns.kdeplot(data = data, x = 'Session_5plus_pages', ax = ax[1][1])
    sns.kdeplot(data = data, x = 'sessions', ax = ax[1][2])

    #plt.show()

    plt.gcf().set_size_inches(20, 10)
    plt.savefig(plots_folder + '/density_session_time_plus.jpg')
    plt.clf()

    ### plot the density of log transformed version of session plus variables

    fig, ax = plt.subplots(nrows = 2, ncols = 3)
    sns.kdeplot(data = data, x = 'Session_1plus_minute_log', ax = ax[0][0])
    sns.kdeplot(data = data, x = 'Session_3plus_minutes_log', ax = ax[0][1])
    sns.kdeplot(data = data, x = 'Session_3plus_pages_log', ax = ax[0][2])
    sns.kdeplot(data = data, x = 'Session_5plus_minutes_log', ax = ax[1][0])
    sns.kdeplot(data = data, x = 'Session_5plus_pages_log', ax = ax[1][1])
    sns.kdeplot(data = data, x = 'sessions_log', ax = ax[1][2])

    #plt.show()

    plt.gcf().set_size_inches(20, 10)
    plt.savefig(plots_folder + '/density_session_time_plus_log.jpg')
    plt.clf()

    ### Plot the count of levels for each categorical variable
    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    
    vars_to_plot = ['device_category', 'non_shopper', 'user_type', 'Lead _Form_submission_str']
    print(ax.ravel())
    for i, sub_ax in enumerate(ax.ravel()):
        print(vars_to_plot[i])

        level_count = data.groupby(vars_to_plot[i])[vars_to_plot[i]].count().to_frame()
        level_count.columns = ['count']
        total = level_count['count'].sum()
        level_count['perc'] = level_count['count']/total
        percentage = level_count['perc']*100
        level_count = level_count.reset_index()
        print(level_count)
        print(total)

        sns.barplot(data = level_count, x = vars_to_plot[i], y = 'count', ax = sub_ax)

        patches = sub_ax.patches
        for i in range(len(patches)):
            x = patches[i].get_x() + patches[i].get_width()/2
            y = patches[i].get_height() + 0.1
            sub_ax.annotate('{:.1f}%'.format(percentage[i]), (x, y), ha='center')

    #plt.show()

    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    plt.gcf().set_size_inches(20, 10)
    plt.savefig(plots_folder + '/categorical_vars_count.jpg')
    plt.clf()


    hue_vars = ['device_category', 'non_shopper', 'user_type', 'Lead _Form_submission_str']

    for h_var in hue_vars:
        fig, ax = plt.subplots(nrows = 2, ncols = 2)

        sns.kdeplot(data = data, x = 'Avg_Session_Duration', hue = h_var, ax = ax[0][0])
        sns.kdeplot(data = data, x = 'avg_time_on_page', hue = h_var, ax = ax[0][1])
        sns.kdeplot(data = data, x = 'Pages_Session', hue = h_var, ax = ax[1][0])
        sns.kdeplot(data = data, x = 'pageviews', hue = h_var, ax = ax[1][1])

        #plt.show()

        plt.gcf().set_size_inches(20, 10)
        plt.savefig(plots_folder + '/session_vars_by_{0}.jpg'.format(h_var))
        plt.clf()


        fig, ax = plt.subplots(nrows = 2, ncols = 2)
        sns.kdeplot(data = data, x = 'Avg_Session_Duration_log', hue = h_var, ax = ax[0][0])
        sns.kdeplot(data = data, x = 'avg_time_on_page_log', hue = h_var, ax = ax[0][1])
        sns.kdeplot(data = data, x = 'Pages_Session_log', hue = h_var, ax = ax[1][0])
        sns.kdeplot(data = data, x = 'pageviews_log', hue = h_var, ax = ax[1][1])

        plt.gcf().set_size_inches(20, 10)
        plt.savefig(plots_folder + '/session_vars_log_by_{0}.jpg'.format(h_var))
        plt.clf()

### Run aggregation of the metric column by categorical variables like
### (device_type, non_shopper, user_type and lead_form_submission)
def aggregation(metric_col, filter_non_zero = False):
    data_file = base_folder + '/CaseSTudy_2_data.xlsx'
    data = pd.read_excel(data_file)

    ### Average Session Duration ###

    #metric_col = 'Avg_Session_Duration'
    #metric_col = 'avg_time_on_page'
    #metric_col = 'Pages_Session'
    #metric_col = 'pageviews'

    filter_non_zero = True

    print('Summary {0}:'.format(metric_col))
    data_avg_metric_non_zero = data[data[metric_col] > 0]
    print('Session Duration Non-Zero:', data_avg_metric_non_zero.shape[0])
    
    ### Aggregation By Device Category
    print('Aggregation (by device category) on full data')
    data_agg_avg_metric_by_device_category = data.groupby(['device_category']).agg(
                                                    count = (metric_col, 'count'),
                                                    mean = (metric_col, 'mean'), 
                                                    median = (metric_col, 'median')
                                                ).reset_index()

    data_agg_avg_metric_by_device_category['perc'] = data_agg_avg_metric_by_device_category['count']/data.shape[0]

    data_agg_quantile_avg_metric_by_device_category = data.groupby(['device_category'])[metric_col].quantile([0.80, 0.90, 0.95]).reset_index()
    data_agg_quantile_avg_metric_by_device_category.columns = ['device_category'] + ['perc', 'value']
    data_agg_quantile_avg_metric_by_device_category = data_agg_quantile_avg_metric_by_device_category.pivot_table(index = 'device_category', columns = ['perc'], values = 'value')
    data_agg_quantile_avg_metric_by_device_category.columns = ['quantile_{0}'.format(c) for c in data_agg_quantile_avg_metric_by_device_category.columns]
    data_agg_quantile_avg_metric_by_device_category = data_agg_quantile_avg_metric_by_device_category.reset_index()

    data_agg_avg_metric_by_device_category = pd.concat((data_agg_avg_metric_by_device_category, data_agg_quantile_avg_metric_by_device_category), axis = 1)
    print(data_agg_avg_metric_by_device_category)

    if (filter_non_zero):
        print('Aggregation (by device category) on non zero sessions data')
        data_agg_avg_metric_non_zero_by_device_category = data_avg_metric_non_zero.groupby(['device_category']).agg(
                                                        count = (metric_col, 'count'),
                                                        mean = (metric_col, 'mean'), 
                                                        median = (metric_col, 'median')
                                                    ).reset_index()

        data_agg_avg_metric_non_zero_by_device_category['perc'] = data_agg_avg_metric_non_zero_by_device_category['count']/data_avg_metric_non_zero.shape[0]

        data_agg_quantile_avg_metric_non_zero_by_device_category = data_avg_metric_non_zero.groupby(['device_category'])[metric_col].quantile([0.80, 0.90, 0.95]).reset_index()
        data_agg_quantile_avg_metric_non_zero_by_device_category.columns = ['device_category'] + ['perc', 'value']
        data_agg_quantile_avg_metric_non_zero_by_device_category = data_agg_quantile_avg_metric_non_zero_by_device_category.pivot_table(index = 'device_category', columns = ['perc'], values = 'value')
        data_agg_quantile_avg_metric_non_zero_by_device_category.columns = ['quantile_{0}'.format(c) for c in data_agg_quantile_avg_metric_non_zero_by_device_category.columns]
        data_agg_quantile_avg_metric_non_zero_by_device_category = data_agg_quantile_avg_metric_non_zero_by_device_category.reset_index()

        data_agg_avg_metric_non_zero_by_device_category = pd.concat((data_agg_avg_metric_non_zero_by_device_category, data_agg_quantile_avg_metric_non_zero_by_device_category), axis = 1)
        print(data_agg_avg_metric_non_zero_by_device_category)

        print('Non zero percentage (by device category)')
        avg_session_non_zero_perc_by_device_category = pd.concat((data_agg_avg_metric_by_device_category[['count']], data_agg_avg_metric_non_zero_by_device_category[['count']]), axis = 1)
        avg_session_non_zero_perc_by_device_category.columns = ['all', 'non_zero']
        avg_session_non_zero_perc_by_device_category['non_zero_perc'] = avg_session_non_zero_perc_by_device_category['non_zero']/avg_session_non_zero_perc_by_device_category['all']
        print(avg_session_non_zero_perc_by_device_category)


    print('==============================')

    ### Aggregation By Non shopper
    print('Aggregation (by non_shopper) on full data')
    data_agg_avg_metric_by_non_shopper = data.groupby(['non_shopper']).agg(
                                                    count = (metric_col, 'count'),
                                                    mean = (metric_col, 'mean'), 
                                                    median = (metric_col, 'median')
                                                ).reset_index()

    data_agg_avg_metric_by_non_shopper['perc'] = data_agg_avg_metric_by_non_shopper['count']/data.shape[0]

    data_agg_quantile_avg_metric_by_non_shopper = data.groupby(['non_shopper'])[metric_col].quantile([0.80, 0.90, 0.95]).reset_index()
    data_agg_quantile_avg_metric_by_non_shopper.columns = ['non_shopper'] + ['perc', 'value']
    data_agg_quantile_avg_metric_by_non_shopper = data_agg_quantile_avg_metric_by_non_shopper.pivot_table(index = 'non_shopper', columns = ['perc'], values = 'value')
    data_agg_quantile_avg_metric_by_non_shopper.columns = ['quantile_{0}'.format(c) for c in data_agg_quantile_avg_metric_by_non_shopper.columns]
    data_agg_quantile_avg_metric_by_non_shopper = data_agg_quantile_avg_metric_by_non_shopper.reset_index()

    data_agg_avg_metric_by_non_shopper = pd.concat((data_agg_avg_metric_by_non_shopper, data_agg_quantile_avg_metric_by_non_shopper), axis = 1)
    print(data_agg_avg_metric_by_non_shopper)

    if (filter_non_zero):
        print('Aggregation (by non_shopper) on non zero sessions data')
        data_agg_avg_metric_non_zero_by_non_shopper = data_avg_metric_non_zero.groupby(['non_shopper']).agg(
                                                        count = (metric_col, 'count'),
                                                        mean = (metric_col, 'mean'), 
                                                        median = (metric_col, 'median')
                                                    ).reset_index()

        data_agg_avg_metric_non_zero_by_non_shopper['perc'] = data_agg_avg_metric_non_zero_by_non_shopper['count']/data_avg_metric_non_zero.shape[0]

        data_agg_quantile_avg_metric_non_zero_by_non_shopper = data_avg_metric_non_zero.groupby(['non_shopper'])[metric_col].quantile([0.80, 0.90, 0.95]).reset_index()
        data_agg_quantile_avg_metric_non_zero_by_non_shopper.columns = ['non_shopper'] + ['perc', 'value']
        data_agg_quantile_avg_metric_non_zero_by_non_shopper = data_agg_quantile_avg_metric_non_zero_by_non_shopper.pivot_table(index = 'non_shopper', columns = ['perc'], values = 'value')
        data_agg_quantile_avg_metric_non_zero_by_non_shopper.columns = ['quantile_{0}'.format(c) for c in data_agg_quantile_avg_metric_non_zero_by_non_shopper.columns]
        data_agg_quantile_avg_metric_non_zero_by_non_shopper = data_agg_quantile_avg_metric_non_zero_by_non_shopper.reset_index()

        data_agg_avg_metric_non_zero_by_non_shopper = pd.concat((data_agg_avg_metric_non_zero_by_non_shopper, data_agg_quantile_avg_metric_non_zero_by_non_shopper), axis = 1)
        print(data_agg_avg_metric_non_zero_by_non_shopper)

        print('Non zero percentage (by non_shopper)')
        avg_session_non_zero_perc_by_non_shopper = pd.concat((data_agg_avg_metric_by_non_shopper[['count']], data_agg_avg_metric_non_zero_by_non_shopper[['count']]), axis = 1)
        avg_session_non_zero_perc_by_non_shopper.columns = ['all', 'non_zero']
        avg_session_non_zero_perc_by_non_shopper['non_zero_perc'] = avg_session_non_zero_perc_by_non_shopper['non_zero']/avg_session_non_zero_perc_by_non_shopper['all']
        print(avg_session_non_zero_perc_by_non_shopper)


    print('==========================================')

    ### Aggregation By Lead Form Submission
    print('Aggregation (by Lead Form Submission) on full data')
    data_agg_avg_metric_by_lead_form = data.groupby(['Lead _Form_submission']).agg(
                                                    count = (metric_col, 'count'),
                                                    mean = (metric_col, 'mean'), 
                                                    median = (metric_col, 'median')
                                                ).reset_index()

    data_agg_avg_metric_by_lead_form['perc'] = data_agg_avg_metric_by_lead_form['count']/data.shape[0]

    data_agg_quantile_avg_metric_by_lead_form = data.groupby(['Lead _Form_submission'])[metric_col].quantile([0.80, 0.90, 0.95]).reset_index()
    data_agg_quantile_avg_metric_by_lead_form.columns = ['Lead _Form_submission'] + ['perc', 'value']
    data_agg_quantile_avg_metric_by_lead_form = data_agg_quantile_avg_metric_by_lead_form.pivot_table(index = 'Lead _Form_submission', columns = ['perc'], values = 'value')
    data_agg_quantile_avg_metric_by_lead_form.columns = ['quantile_{0}'.format(c) for c in data_agg_quantile_avg_metric_by_lead_form.columns]
    data_agg_quantile_avg_metric_by_lead_form = data_agg_quantile_avg_metric_by_lead_form.reset_index()

    data_agg_avg_metric_by_lead_form = pd.concat((data_agg_avg_metric_by_lead_form, data_agg_quantile_avg_metric_by_lead_form), axis = 1)
    print(data_agg_avg_metric_by_lead_form)

    if (filter_non_zero):
        print('Aggregation (by lead_form) on non zero sessions data')
        data_agg_avg_metric_non_zero_by_lead_form = data_avg_metric_non_zero.groupby(['Lead _Form_submission']).agg(
                                                        count = (metric_col, 'count'),
                                                        mean = (metric_col, 'mean'), 
                                                        median = (metric_col, 'median')
                                                    ).reset_index()

        data_agg_avg_metric_non_zero_by_lead_form['perc'] = data_agg_avg_metric_non_zero_by_lead_form['count']/data_avg_metric_non_zero.shape[0]

        data_agg_quantile_avg_metric_non_zero_by_lead_form = data_avg_metric_non_zero.groupby(['Lead _Form_submission'])[metric_col].quantile([0.80, 0.90, 0.95]).reset_index()
        data_agg_quantile_avg_metric_non_zero_by_lead_form.columns = ['Lead _Form_submission'] + ['perc', 'value']
        data_agg_quantile_avg_metric_non_zero_by_lead_form = data_agg_quantile_avg_metric_non_zero_by_lead_form.pivot_table(index = 'Lead _Form_submission', columns = ['perc'], values = 'value')
        data_agg_quantile_avg_metric_non_zero_by_lead_form.columns = ['quantile_{0}'.format(c) for c in data_agg_quantile_avg_metric_non_zero_by_lead_form.columns]
        data_agg_quantile_avg_metric_non_zero_by_lead_form = data_agg_quantile_avg_metric_non_zero_by_lead_form.reset_index()

        data_agg_avg_metric_non_zero_by_lead_form = pd.concat((data_agg_avg_metric_non_zero_by_lead_form, data_agg_quantile_avg_metric_non_zero_by_lead_form), axis = 1)
        print(data_agg_avg_metric_non_zero_by_lead_form)

        print('Non zero percentage (by lead_form)')
        avg_session_non_zero_perc_by_lead_form = pd.concat((data_agg_avg_metric_by_lead_form[['count']], data_agg_avg_metric_non_zero_by_lead_form[['count']]), axis = 1)
        avg_session_non_zero_perc_by_lead_form.columns = ['all', 'non_zero']
        avg_session_non_zero_perc_by_lead_form['non_zero_perc'] = avg_session_non_zero_perc_by_lead_form['non_zero']/avg_session_non_zero_perc_by_lead_form['all']
        print(avg_session_non_zero_perc_by_lead_form)

    print('==========================================')

    ### Aggregation By User Type
    print('Aggregation (by user_type) on full data')
    data_agg_avg_metric_by_user_type = data.groupby(['user_type']).agg(
                                                    count = (metric_col, 'count'),
                                                    mean = (metric_col, 'mean'), 
                                                    median = (metric_col, 'median')
                                                ).reset_index()

    data_agg_avg_metric_by_user_type['perc'] = data_agg_avg_metric_by_user_type['count']/data.shape[0]

    data_agg_quantile_avg_metric_by_user_type = data.groupby(['user_type'])[metric_col].quantile([0.80, 0.90, 0.95]).reset_index()
    data_agg_quantile_avg_metric_by_user_type.columns = ['user_type'] + ['perc', 'value']
    data_agg_quantile_avg_metric_by_user_type = data_agg_quantile_avg_metric_by_user_type.pivot_table(index = 'user_type', columns = ['perc'], values = 'value')
    data_agg_quantile_avg_metric_by_user_type.columns = ['quantile_{0}'.format(c) for c in data_agg_quantile_avg_metric_by_user_type.columns]
    data_agg_quantile_avg_metric_by_user_type = data_agg_quantile_avg_metric_by_user_type.reset_index()

    data_agg_avg_metric_by_user_type = pd.concat((data_agg_avg_metric_by_user_type, data_agg_quantile_avg_metric_by_user_type), axis = 1)
    print(data_agg_avg_metric_by_user_type)

    if (filter_non_zero):
        print('Aggregation (by user_type) on non zero sessions data')
        data_agg_avg_metric_non_zero_by_user_type = data_avg_metric_non_zero.groupby(['user_type']).agg(
                                                        count = (metric_col, 'count'),
                                                        mean = (metric_col, 'mean'), 
                                                        median = (metric_col, 'median')
                                                    ).reset_index()

        data_agg_avg_metric_non_zero_by_user_type['perc'] = data_agg_avg_metric_non_zero_by_user_type['count']/data_avg_metric_non_zero.shape[0]

        data_agg_quantile_avg_metric_non_zero_by_user_type = data_avg_metric_non_zero.groupby(['user_type'])[metric_col].quantile([0.80, 0.90, 0.95]).reset_index()
        data_agg_quantile_avg_metric_non_zero_by_user_type.columns = ['user_type'] + ['perc', 'value']
        data_agg_quantile_avg_metric_non_zero_by_user_type = data_agg_quantile_avg_metric_non_zero_by_user_type.pivot_table(index = 'user_type', columns = ['perc'], values = 'value')
        data_agg_quantile_avg_metric_non_zero_by_user_type.columns = ['quantile_{0}'.format(c) for c in data_agg_quantile_avg_metric_non_zero_by_user_type.columns]
        data_agg_quantile_avg_metric_non_zero_by_user_type = data_agg_quantile_avg_metric_non_zero_by_user_type.reset_index()

        data_agg_avg_metric_non_zero_by_user_type = pd.concat((data_agg_avg_metric_non_zero_by_user_type, data_agg_quantile_avg_metric_non_zero_by_user_type), axis = 1)
        print(data_agg_avg_metric_non_zero_by_user_type)

        print('Non zero percentage (by user_type)')
        avg_session_non_zero_perc_by_user_type = pd.concat((data_agg_avg_metric_by_user_type[['count']], data_agg_avg_metric_non_zero_by_user_type[['count']]), axis = 1)
        avg_session_non_zero_perc_by_user_type.columns = ['all', 'non_zero']
        avg_session_non_zero_perc_by_user_type['non_zero_perc'] = avg_session_non_zero_perc_by_user_type['non_zero']/avg_session_non_zero_perc_by_user_type['all']
        print(avg_session_non_zero_perc_by_user_type)


### scratchpad function to check the data:
### Not used in generating any output or insights
def user_count():
    data_file = base_folder + '/CaseSTudy_2_data_no_dup.xlsx'
    data = pd.read_excel(data_file)

    data['Visitor_Identifier'] = data['Visitor_Identifier'].astype('int64').astype('str')

    print(data)

    n = data.shape[0]
    num_unique = len(data['Visitor_Identifier'].unique())

    print('N:', n)
    print('num unique:', num_unique)

    print(n - num_unique)

    print(data['Lead _Form_submission'].sum())

    user_count = data.groupby(['Visitor_Identifier'])['Visitor_Identifier'].count().to_frame()
    user_count.columns = ['user_count']
    user_count = user_count.reset_index()
    print(user_count)

    data = pd.merge(data, user_count, on = 'Visitor_Identifier', how = 'inner')
    print(data)

    print(data.groupby(['user_count', 'user_type'])['Visitor_Identifier'].count())
    print(data.groupby(['user_count', 'Lead _Form_submission'])['Visitor_Identifier'].count())

    data_no_dup = copy(data)

    data_no_dup['row_num'] = data.groupby(['Visitor_Identifier']).cumcount() + 1
    print(data_no_dup)

### function to remove repeating visitor id
### Assumption is that the data is aggregated at the vistor id level
### and should have one rcord per visitor
def data_drop_dup():

    data_file = base_folder + '/CaseSTudy_2_data.xlsx'
    data = pd.read_excel(data_file)

    data['Visitor_Identifier'] = data['Visitor_Identifier'].astype('int64').astype('str')

    data_no_dup = copy(data)

    data_no_dup['row_num'] = data.groupby(['Visitor_Identifier']).cumcount() + 1

    max_row_num = data_no_dup.groupby(['Visitor_Identifier'])['row_num'].max().to_frame()
    max_row_num.columns = ['max_row_num']
    max_row_num = max_row_num.reset_index()

    data_no_dup = pd.merge(data_no_dup, max_row_num, on = 'Visitor_Identifier', how = 'inner')
    data_no_dup = data_no_dup[data_no_dup['row_num'] == data_no_dup['max_row_num']]

    data_no_dup = data_no_dup.drop(columns = ['row_num', 'max_row_num'])

    print(data_no_dup)

    data_no_dup.to_excel(base_folder + '/CaseSTudy_2_data_no_dup.xlsx', index = None)

def general_metrics():

    data_file = base_folder + '/CaseSTudy_2_data_no_dup.xlsx'
    data = pd.read_excel(data_file)

    n = data.shape[0]
    print('Percentage visitors with Avg Session Duration of 0:', data[data['Avg_Session_Duration'] == 0].shape[0]/n)

    data_session_duration_nonzero = data[data['Avg_Session_Duration'] != 0]

    print('Percentage of visitors having non-zero session duration with 1 visit: ', 
            data_session_duration_nonzero[data_session_duration_nonzero['sessions'] == 1].shape[0]/data_session_duration_nonzero.shape[0])




def main():
    #distribution_plots()


    ### Run some aggregations on these metrics by categorical variables.
    aggregation('Avg_Session_Duration', True)
    aggregation('avg_time_on_page', False)
    aggregation('Pages_Session', False)
    aggregation('pageviews', False)
    
    
    #user_count()
    #data_drop_dup()
    #general_metrics()


if __name__ == '__main__':
    main()