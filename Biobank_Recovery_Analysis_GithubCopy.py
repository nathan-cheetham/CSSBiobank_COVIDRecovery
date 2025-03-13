# -*- coding: utf-8 -*-
"""
CSS Biobank COVID-19 recovery analysis, based mainly on analysis of follow-up 2022 Long Covid Questionnaire
"""

import numpy as np
import pandas as pd
import copy
from stepmix.stepmix import StepMix
from sklearn.model_selection import GridSearchCV, ParameterGrid
from datetime import datetime
from time import time
import statsmodels.api as sm
from statsmodels.genmod.families.links import logit, identity, log
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.ticker import StrMethodFormatter, NullFormatter, FormatStrFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
plt.rc("font", size=12)
import scipy as sp
from scipy.stats import norm
from scipy import stats
from scipy.stats import chi2_contingency 
from scipy.stats import kruskal 
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score, explained_variance_score, r2_score
import pymannkendall as mk
import seaborn as sns
import sys
sys.path.append("C:/Users/k2143494/OneDrive - King's College London/Documents/nc_scripts") # add nc_scripts folder to path so modules contained within can be imported
import processing_functions as pf
import analysis_functions as af
import Biobank_Recovery_Codebook

sns.set_style("whitegrid")

export_csv = 0

# Set which stringency limit on logging frequency to use: '7' or '14' 
stringency = '14' 

do_correlation = 0

codebook = {}

# list of missing data strings
missing_data_values = [np.nan, 'NaN','nan'] 

#%% Define functions
# -----------------------------------------------------------------------------
### Add dummy variable fields to dataframe generated from un-ordered categoricals
def categorical_to_dummy(df, variable_list_categorical):
    """Create dummy variables from un-ordered categoricals"""
    # Create dummy variables
    dummy_var_list_full = []
    for var in variable_list_categorical:
        df[var] = df[var].fillna('NaN') # fill NaN with 'No data' so missing data can be distinguished from 0 results
        cat_list ='var'+'_'+var # variable name
        cat_list = pd.get_dummies(df[var], prefix=var) # create binary variable of category value
        df = df.join(cat_list) # join new column to dataframe
    
    return df

# -----------------------------------------------------------------------------
### Generate list of categorical dummy variables from original fieldname, deleting original fieldname and deleting reference variable using reference dummy variable list
def generate_dummy_list(original_fieldname_list, full_fieldname_list, reference_fieldname_list, delete_reference):
    """ Generate list of categorical dummy variables from original fieldname, deleting original fieldname. Option to also delete reference variable using reference dummy variable list (delete_reference = 'yes') """
    dummy_list = []
    for var in original_fieldname_list:
        print(var)
        var_matching_all = [variable_name for variable_name in full_fieldname_list if var in variable_name]
        # drop original variable
        var_matching_all.remove(var)
        if delete_reference == 'yes':
            # drop reference variable
            var_matching_reference = [variable_name for variable_name in reference_fieldname_list if var in variable_name][0]
            var_matching_all.remove(var_matching_reference)
        # add to overall list
        dummy_list += var_matching_all
    
    return dummy_list

# -----------------------------------------------------------------------------
# Do manual winsorisation as scipy winsorize function has bugs (treats nan as high number rather than ignoring) which means upper end winsorisation doesn't work
def winsorization(data, winsorization_limits, winsorization_col_list, set_manual_limits, manual_limit_list):
    for n in range(0, len(winsorization_col_list), 1): 
        col = winsorization_col_list[n]
        # Create copy columns 
        data[col + '_winsorised'] = data[col].copy()
        
        if set_manual_limits == 'yes':
            winsorize_lower = manual_limit_list[n][0]
            winsorize_upper = manual_limit_list[n][1]
        else:
            # Calculate percentile
            winsorize_lower = data[col].quantile(winsorization_limits[0])
            winsorize_upper = data[col].quantile(winsorization_limits[1])
            
        print('lower limit = ' + str(winsorize_lower))
        print('higher limit = ' + str(winsorize_upper))
        # Replace lower and upper values with limited values
        data.loc[(data[col] < winsorize_lower), col + '_winsorised'] = winsorize_lower
        data.loc[(data[col] > winsorize_upper), col + '_winsorised'] = winsorize_upper
    return data


# -----------------------------------------------------------------------------
### Run logistic regression model with HC3 robust error, producing summary dataframe  
def sm_logreg_simple_HC3(x_data, y_data, CI_alpha, do_robust_se, use_weights, weight_data, do_poisson):
    """ Run logistic regression model with HC3 robust error, producing summary dataframe """
    # Add constant - default for sklearn but not statsmodels
    x_data = sm.add_constant(x_data) 
    
    # Add weight data to x_data if weights specified
    if use_weights == 'yes':
        x_data['weight'] = weight_data
        
    # Set model parameters
    max_iterations = 2000
    solver_method = 'newton' # use default
    # model = sm.Logit(y_data, x_data, use_t = True) # Previous model - same results. Replaced with more general construction, as GLM allows weights to be included. 
    
    # Also run model on test and train split to assess predictive power
    # Generate test and train split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, stratify = y_data, test_size = 0.25, random_state = 0)
    
    # Save weight data in x_train and the drop weight data 
    if use_weights == 'yes':
        weight_data_train = np.asarray(x_train['weight'].copy())
        # drop weight columns
        x_data = x_data.drop(columns = ['weight'])
        x_train = x_train.drop(columns = ['weight'])
        x_test = x_test.drop(columns = ['weight'])
    
    
    
    # Set up overall and test-train models
    if use_weights == 'yes':
        model = sm.GLM(y_data, x_data, 
                       var_weights = weight_data, 
                       family = sm.families.Binomial(),
                       link=logit)
        model_testtrain = sm.GLM(y_train, x_train, 
                       var_weights = weight_data_train, 
                       family = sm.families.Binomial(),
                       link=logit)
        if do_poisson == 'yes':
            model = sm.GLM(y_data, x_data, 
                       var_weights = weight_data,
                       family = sm.families.Poisson(), 
                       link=log)
            model_testtrain = sm.GLM(y_train, x_train,
                       var_weights = weight_data_train,
                       family = sm.families.Poisson(),
                       link=log)
    else:
        model = sm.GLM(y_data, x_data, 
                       family = sm.families.Binomial(), 
                       link=logit)
        model_testtrain = sm.GLM(y_train, x_train,
                       family = sm.families.Binomial(),
                       link=logit)
        if do_poisson == 'yes':
            model = sm.GLM(y_data, x_data, 
                       family = sm.families.Poisson(), 
                       link=log)
            model_testtrain = sm.GLM(y_train, x_train,
                       family = sm.families.Poisson(),
                       link=log)
    # Fit model
    if do_robust_se == 'HC3':
        model_fit = model.fit(maxiter = max_iterations, 
                          method = solver_method,  
                          cov_type='HC3',
                          use_t=True)
        model_testtrain_fit = model_testtrain.fit(maxiter = max_iterations, 
                          method = solver_method,  
                          cov_type='HC3',
                          use_t=True)
    else:
        model_fit = model.fit(maxiter = max_iterations, 
                              method = solver_method, 
                              use_t=True)
        model_testtrain_fit = model_testtrain.fit(maxiter = max_iterations, 
                              method = solver_method, 
                              use_t=True) 
    
    # print(model_fit.summary())
   
    # Calculate AUC and explained variance score of model
    y_prob = model_testtrain_fit.predict(x_test)
    if np.isnan(np.min(y_prob)) == False:
        model_auc = roc_auc_score(y_test, y_prob)
        model_explained_variance = explained_variance_score(y_test, y_prob)
        model_r2 = r2_score(y_test, y_prob)
    else:
        print(y_prob)
        model_auc = 0 # for when AUC failed - e.g. due to non-convergence of model
        model_explained_variance = np.nan
        model_r2 = np.nan
        
    # Extract coefficients and convert to Odds Ratios
    sm_coeff = model_fit.params
    sm_se = model_fit.bse
    sm_pvalue = model_fit.pvalues
    sm_coeff_CI = model_fit.conf_int(alpha=CI_alpha)
    sm_OR = np.exp(sm_coeff)
    sm_OR_CI = np.exp(sm_coeff_CI)
    
    # Create dataframe summarising results
    sm_summary = pd.DataFrame({'Variable': sm_coeff.index,
                               'Coefficients': sm_coeff,
                               'Standard Error': sm_se,
                               'P-value': sm_pvalue,
                               'Coefficient C.I. (lower)': sm_coeff_CI[0],
                               'Coefficient C.I. (upper)': sm_coeff_CI[1],
                               'Odds ratio': sm_OR,
                               'OR C.I. (lower)': sm_OR_CI[0],
                               'OR C.I. (upper)': sm_OR_CI[1],
                               'OR C.I. error (lower)': np.abs(sm_OR - sm_OR_CI[0]),
                               'OR C.I. error (upper)': np.abs(sm_OR - sm_OR_CI[1]),
                                })
    sm_summary = sm_summary.reset_index(drop = True)
    
    # Add total number of individuals in given model
    sm_summary['total_count_n'] = len(x_data)

    # Add number of observations for given variable in input and outcome datasets
    x_data_count = x_data.sum()
    x_data_count.name = "group_count"
    sm_summary = pd.merge(sm_summary,x_data_count, how = 'left', left_on = 'Variable', right_index = True)
    
    # join x_data and y_data
    x_y_data = x_data.copy()
    x_y_data['y_data'] = y_data
    # Count observation where y_data = 1
    y_data_count = x_y_data[x_y_data['y_data'] == 1].sum()
    y_data_count.name = "outcome_count"
    sm_summary = pd.merge(sm_summary,y_data_count, how = 'left', left_on = 'Variable', right_index = True)
    
    # Highlight variables where confidence intervals are both below 1 or both above 1
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] > 1.0)
                        & (sm_summary['OR C.I. (upper)'] > 1.0)
                        & (sm_summary['P-value'] < 0.05)
                        ,'Significance'] = 'Significant (OR > 1), *, p < 0.05'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] > 1.0)
                        & (sm_summary['OR C.I. (upper)'] > 1.0)
                        & (sm_summary['P-value'] < 0.01)
                        ,'Significance'] = 'Significant (OR > 1), **, p < 0.01'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] > 1.0)
                        & (sm_summary['OR C.I. (upper)'] > 1.0)
                        & (sm_summary['P-value'] < 0.001)
                        ,'Significance'] = 'Significant (OR > 1), ***, p < 0.001'
    
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] < 1.0)
                        & (sm_summary['OR C.I. (upper)'] < 1.0)
                        & (sm_summary['P-value'] < 0.05)
                        ,'Significance'] = 'Significant (OR < 1), *, p < 0.05'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] < 1.0)
                        & (sm_summary['OR C.I. (upper)'] < 1.0)
                        & (sm_summary['P-value'] < 0.01)
                        ,'Significance'] = 'Significant (OR < 1), **, p < 0.01'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] < 1.0)
                        & (sm_summary['OR C.I. (upper)'] < 1.0)
                        & (sm_summary['P-value'] < 0.001)
                        ,'Significance'] = 'Significant (OR < 1), ***, p < 0.001'
        
    return sm_summary, model_fit, model_auc, model_explained_variance, model_r2

# -----------------------------------------------------------------------------
### Calculate correlation between variables
def calculate_correlation(data, data_cols, corr_thresh, mask_on, dictionary, annotate):
    """ Function to calculate and visualise correlation within dataset - designed to test correlation between cognition task metrics """
    # Filter for relevant metrics
    task_data = data[data_cols].copy()
    
    # Calculate correlation between variable values 
    data_correlation = task_data.corr()
    
    if dictionary != 'no':
        data_correlation = data_correlation.rename(index = dictionary)
        data_correlation = data_correlation.rename(columns = dictionary)
    
    # Flatten and filter to identify correlations larger than a threshold and decide which features to eliminate
    data_correlation_flat = data_correlation.stack().reset_index()
    data_correlation_above_thresh = data_correlation_flat[(data_correlation_flat[0].abs() >= corr_thresh)
                                                              & (data_correlation_flat[0].abs() < 1)]
    
    # set up mask so only bottom left part of matrix is shown
    mask = np.zeros_like(data_correlation)
    mask[np.triu_indices_from(mask)] = True
    
    plt.figure()
    if mask_on == 'yes':
        # sns.heatmap(data_correlation, cmap = 'YlGnBu', linewidths=.5, mask = mask, center = 0)
        plt.figure(figsize = (15,12))
        if annotate == 'yes':
            sns.heatmap(data_correlation, 
                        cmap = 'bwr', # bwr YlGnBu
                        linewidths=.5, 
                        mask = mask,
                        annot=True, fmt=".2f",
                        center = 0, vmin = -1, vmax = 1
                        )
        else: 
            sns.heatmap(data_correlation, 
                        cmap = 'bwr', # bwr YlGnBu
                        linewidths=.5, 
                        mask = mask,
                        # annot=True, fmt=".2f",
                        center = 0, vmin = -1, vmax = 1
                        )
    else:
        plt.figure(figsize = (15,12))
        if annotate == 'yes':
            sns.heatmap(data_correlation, 
                        cmap = 'bwr', # bwr YlGnBu
                        linewidths=.5, 
                        annot=True, fmt=".2f",
                        center = 0, vmin = -1, vmax = 1
                        )
        else: 
            sns.heatmap(data_correlation, 
                        cmap = 'bwr', # bwr YlGnBu
                        linewidths=.5, 
                        # annot=True, fmt=".2f",
                        center = 0, vmin = -1, vmax = 1
                        )
    plt.title('Correlation matrix')
        
    return data_correlation, data_correlation_flat


# -----------------------------------------------------------------------------
# Function to run series of logistic regression models in sequence
def run_logistic_regression_models(data, data_full_col_list, logreg_model_var_list, outcome_var, use_weights, weight_var, filter_missing, plot_model, do_poisson):
    """ Function to run logistic regression models, given lists of categorical and continuous input variables """
    model_input_list = []
    model_auc_list= []
    model_summary_list = []
    model_fit_list = []
    for sublist in logreg_model_var_list:
        var_continuous = sublist[0]
        var_categorical = sublist[1]
        var_exposure = sublist[2] # identify exposure variable being tested in model
        
        print('Exposure variable: ' + var_exposure)
        
        # Filter out missing or excluded data
        print('Individuals before filtering: ' + str(data.shape[0]))
        data_filterformodel = data.copy()
        
        if filter_missing == 'yes':
            # Filter using original categorical variables
            for col in var_categorical:
                    data_filterformodel = data_filterformodel[~(data_filterformodel[col].isin(missing_data_values))]
            print('Individuals after filtering: ' + str(data_filterformodel.shape[0]))
            # Filter using original continuous variables
            for col in var_continuous:
                    data_filterformodel = data_filterformodel[~(data_filterformodel[col].isin(missing_data_values))]
            print('Individuals after filtering: ' + str(data_filterformodel.shape[0]))
            
            # for col in input_var_control_test:
            #         data_filterformodel = data_filterformodel[~(data_filterformodel[col].isin(missing_data_values))]
            # print('Individuals after filtering: ' + str(data_filterformodel.shape[0]))
        
        # Generate list of dummy fields for complete fields
        var_categorical_dummy = generate_dummy_list(original_fieldname_list = var_categorical, 
                                                         full_fieldname_list = data_full_col_list, 
                                                         reference_fieldname_list = cols_categorical_reference,
                                                         delete_reference = 'yes')
        print('Before dropping MISSING DATA and EMPTY dummy cols: ')
        print(var_categorical_dummy)
        if filter_missing == 'yes':
            # Drop dummy columns with name ending in missing data values (categorical only)
            var_categorical_dummy_copy = var_categorical_dummy.copy()
            for col in var_categorical_dummy_copy:
                print(col)
                for missing_val in missing_data_values[1:]:
                    if '_'+str(missing_val) in col:
                        print('remove missing data dummy: ' + col)
                        var_categorical_dummy.remove(col)
            # print('After dropping MISSING DATA dummy cols: ')
            # print(var_categorical_dummy)
        
        # Drop dummy columns where sum of column = 0, 1 or 2 - i.e. no-one or low numbers from particular group - can cause 'Singular matrix' error when running model
        var_categorical_dummy_copy = var_categorical_dummy.copy()
        for col in var_categorical_dummy_copy: 
            # print(col)
            if data_filterformodel[col].sum() <= 2:
                print('remove empty/low number (<=2) dummy: ' + col)
                var_categorical_dummy.remove(col)
        
        # Drop dummy columns where no observations of outcome in group in column = 0 - i.e. no observations - can cause 'Singular matrix' error when running model
        var_categorical_dummy_copy = var_categorical_dummy.copy()
        for col in var_categorical_dummy_copy: 
            if data_filterformodel[(data_filterformodel[outcome_var] == 1)][col].sum() <= 1:
                print('remove dummy with <=1 observations of outcome: ' + col)
                var_categorical_dummy.remove(col)
    
        print('After dropping MISSING DATA and EMPTY dummy cols: ')
        print(var_categorical_dummy)
        
        # Set variables to go into model
        input_var_control_test = var_continuous + var_categorical_dummy
               
        
        model_input = str(var_continuous + var_categorical)
        model_input_list.append(model_input)
        print('model input variables: ' + model_input)
        print('model input variables (dummy): ' + str(input_var_control_test))
        
        # generate x dataset for selected control and test dummy variables only
        logreg_data_x = data_filterformodel[input_var_control_test].reset_index(drop=True) # create input variable tables for models 
        # generate y datasets from selected number of vaccinations and outcome of interest
        logreg_data_y = data_filterformodel[outcome_var].reset_index(drop=True) # set output variable
        
        if use_weights == 'yes':
            logreg_data_weight = data_filterformodel[weight_var].reset_index(drop=True) # filter for weight variable
            # Do logistic regression (stats models) of control + test variables
            sm_summary, model_fit, model_auc, model_explained_variance, model_r2 = sm_logreg_simple_HC3(x_data = logreg_data_x, y_data = logreg_data_y, 
                                                     CI_alpha = 0.05, do_robust_se = 'HC3',
                                                     use_weights = use_weights, weight_data = np.asarray(logreg_data_weight),
                                                     do_poisson = do_poisson)
        else:
            sm_summary, model_fit, model_auc, model_explained_variance, model_r2 = sm_logreg_simple_HC3(x_data = logreg_data_x, y_data = logreg_data_y, 
                                                     CI_alpha = 0.05, do_robust_se = 'HC3',
                                                     use_weights = '', weight_data = '',
                                                     do_poisson = do_poisson)
            
        sm_summary['model_input'] = model_input
        sm_summary['var_exposure'] = var_exposure
        sm_summary['outcome_variable'] = outcome_var
        sm_summary['model_auc'] = model_auc
        sm_summary['model_explained_variance'] = model_explained_variance
        sm_summary['model_r2'] = model_r2
        model_summary_list.append(sm_summary)
        model_fit_list.append(model_fit)
        
        # Print predictive power
        model_auc_list.append(model_auc) 
        print ('AUC: ' + str(model_auc))
        
        # Plot odds ratios
        if plot_model == 'yes':
            fig = af.plot_OR_w_conf_int(sm_summary, 'Variable', 'Odds ratio', ['OR C.I. error (lower)','OR C.I. error (upper)'], 'Odds Ratio', ylims = [], titlelabel = '')
    
    # -----------------------------------------------------------------------------
    # Combine model results tables together
    model_results_summary = pd.concat(model_summary_list)
    model_auc_summary = pd.DataFrame({'model_input':model_input_list,
                                      'model_auc':model_auc_list,})
    
    return model_results_summary, model_auc_summary, model_fit_list



# -----------------------------------------------------------------------------
### For categorical variables, do chi-squared univariate association test and plot cross-tabulation
def categorical_variables_chi_square(data, input_var_categorical, outcome_var, drop_missing, plot_crosstab, print_vars):
    """ Perform chi-square test on cross-tabulation between outcome variable and each categorical variable in input list. Also print cross-tab as proportion in stacked bar chart """
    p_value_list = []
    p_value_missing_dropped_list = []
    var_list_chisquare = []
    for var_cat in input_var_categorical:
        if print_vars == 'yes':
            print(var_cat + ' x ' + outcome_var)
        # Cross tab of frequencies
        crosstab = pd.crosstab(data[var_cat], data[outcome_var], margins = False)            
        # Generate chi-squared statistic
        chi2, p, dof, ex = chi2_contingency(crosstab, correction=False)
        # Save chi-squared results to list
        p_value_list.append(p)
        
        # Drop missing data
        if drop_missing == 'yes':
            crosstab = crosstab.reset_index()
            crosstab = crosstab[~(crosstab[var_cat].isin(missing_data_values))]
            crosstab = crosstab.set_index(var_cat)
                
            # Generate chi-squared statistic
            chi2, p, dof, ex = chi2_contingency(crosstab, correction=False)
            # Save chi-squared results to list
            p_value_missing_dropped_list.append(p)
        
        if plot_crosstab == 'yes':
            # plot proportion
            af.crosstab_plots(crosstab,var_cat,outcome_var)
    
    # Create dataframe showing chi-square test results
    chisquare_results = pd.DataFrame({'Variable':input_var_categorical,
                                      'Chi-squared p-value (with missing)':p_value_list,                                      
                                      })
    if drop_missing == 'yes':
        chisquare_results['Chi-squared p-value (no missing)'] = p_value_missing_dropped_list
    
    return chisquare_results


# -----------------------------------------------------------------------------
# 1 SERIES scatter plot
def plot_OR_w_conf_int(data1, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, xlims, ylims, titlelabel, width, height, y_pos_manual, color_list, fontsize, invert_axis, x_logscale, legend_offset, x_major_tick, x_minor_tick, poisson_reg, bold):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    
    if y_pos_manual == 'yes':
        data1['x_manual'] = (data1['y_pos_manual'])
    else:
        data1['x_manual'] = (np.arange(len(data1[x_fieldname])))
    
    # plot scatter points
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[0], s = 5, label = plot1_label, figsize=(width,height))
    # plot error bars
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 4, label = None, fmt = 'none', color = color_list[0])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.yticks(data1['x_manual'], data1[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    if poisson_reg == 'yes':
        ax.set_xlabel('Relative risk ratio')
    else:
        ax.set_xlabel('Odds ratio')
        
    ax.yaxis.label.set_visible(False) # hide y axis title
    ax.set_title(titlelabel)
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way
    
    if x_logscale == 'yes':
        ax.set_xscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # # Add secondary y axis at top to show raw coefficient before conversion to OR
        # secax_y = ax.twinx()
        # secax_y.spines.right.set_position(("axes", 0))
        # secax_y.spines['left'].set_visible(False)
        # secax_y.set_yticks([0.5])
        # secax_y.set_yticklabels(['test'])
        # for tick in secax_y.yaxis.get_majorticklabels():
        #     tick.set_horizontalalignment("right")
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twiny()
        secax.set_xscale('linear')
        secax.set_xlim(np.log(xlims[0]), np.log(xlims[1]))
        secax.grid(False, which="both", axis = 'x')
        secax.set_xlabel('Coefficient')
        secax.tick_params(axis='x', which='major', labelsize=10)
        secax.set_xlabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', labelsize=8.2)
        
        xlim_thresh = 3.5
        if xlims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',                                   
                                      '1.1','','1.3','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif xlims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '1.5','2', 
                                      '', '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.5, 2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        
        ax.xaxis.set_minor_formatter(x_formatter)
        ax.xaxis.set_minor_locator(x_locator)
                
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        
    if x_major_tick > 0:
        secax.xaxis.set_major_locator(MultipleLocator(x_major_tick))
    if x_minor_tick > 0:
        secax.xaxis.set_minor_locator(MultipleLocator(x_minor_tick))
        
    if bold == 1:
        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')
    
    return data1

# -----------------------------------------------------------------------------
# 2 SERIES scatter plot
def plot_OR_w_conf_int_2plots(data1, data2, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, x_logscale, x_major_tick, x_minor_tick, poisson_reg):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
        
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['y_pos_manual']*scalar) + offset 
        data2['x_manual'] = (data2['y_pos_manual']*scalar) - offset
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + offset 
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar) - offset
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(width,height))
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    # Plot 2
    ax = data2.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[1], s = 5, label = plot2_label, ax = ax)
    error_bar2 = ax.errorbar(y = data2['x_manual'], x = data2[y_fieldname], xerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.yticks(data1['x_manual'] - (offset), data1[x_fieldname]) # set labels manually
    # plt.yticks(data2['x_manual'] + (offset), data2[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    
    if poisson_reg == 'yes':
        ax.set_xlabel('Relative risk ratio')
    else:
        ax.set_xlabel('Odds ratio')
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way

    if x_logscale == 'yes':
        ax.set_xscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twiny()
        secax.set_xscale('linear')
        secax.set_xlim(np.log(xlims[0]), np.log(xlims[1]))
        secax.grid(False, which="both", axis = 'x')
        secax.set_xlabel('Coefficient')
        secax.tick_params(axis='x', which='major', labelsize=10)
        secax.set_xlabel('Coefficient')
        
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', labelsize=8.2)
        
        xlim_thresh = 3.5
        if xlims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',                                   
                                      '1.1','','1.3','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif xlims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '1.5','2', 
                                      '3', '4', '', '6', '', '8', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.5, 2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        ax.xaxis.set_minor_formatter(x_formatter)
        ax.xaxis.set_minor_locator(x_locator)
                    
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    
    if x_major_tick > 0:
        secax.xaxis.set_major_locator(MultipleLocator(x_major_tick))
    if x_minor_tick > 0:
        secax.xaxis.set_minor_locator(MultipleLocator(x_minor_tick))
    
    return data1, data2


# -----------------------------------------------------------------------------
# 1 SERIES bar plot for crude rates
def plot_bar_1series(dataset_1, model_results_combined_filter, x_fieldname, y_fieldname, xlims, ylims, width, height, y_pos_manual, color_list, fontsize, invert_axis, offset, scalar, outcome_var_categorical, outcome_var_dummy, hide_ticks, xlabel):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    
    # Get unique dummy variable names and calculate counts and proportions
    # Dataset 1
    data1 = pd.DataFrame(model_results_combined_filter['Variable']).drop_duplicates()
    for var in data1['Variable']:
        print(var)
        data1['total_count_n'] = dataset_1.shape[0]
        data1.loc[(data1['Variable'] == var), 'group_count'] = dataset_1[~(dataset_1[outcome_var_categorical].isin(missing_data_values))][var].sum(axis = 0)
        data1.loc[(data1['Variable'] == var), 'outcome_count'] = dataset_1[(dataset_1[outcome_var_dummy] == 1)][var].sum(axis = 0)
        data1['group_prop'] = data1['group_count'] / data1['total_count_n']
        data1['outcome_prop'] = data1['outcome_count'] / data1['group_count']
        data1['y_pos_manual'] = data1['Variable'].map(Biobank_Recovery_Codebook.dictionary['y_pos_manual_with_reference_v2'])
    data1['Variable_tidy'] = data1['Variable'].map(Biobank_Recovery_Codebook.dictionary['variable_tidy_V2'])
    data1 = data1[~(data1['y_pos_manual'].isnull())] # drop rows without numbers - not for plotting
    
    # Drop variables which are continuous - not for plotting - find by having 'unit:' or 'Per +1' in Variable_tidy string
    data1 = data1[~((data1['Variable_tidy'].str.contains('unit:', na=False))
                     | (data1['Variable_tidy'].str.contains('Per +1', na=False))
                     | (data1['Variable_tidy'].str.contains('Absence', na=False))
                     )] 
    
    if y_pos_manual == 'yes':
        data1['x_manual'] = (data1['y_pos_manual'])
    else:
        data1['x_manual'] = (np.arange(len(data1[x_fieldname])))
    
    # plot bars
    offset = offset
    fig, ax = plt.subplots(figsize=(width,height))
    # plt.figure(figsize=(width,height))
    ax.barh(data1['x_manual'], data1[y_fieldname], height = offset*scalar, color = color_list[0], align='center')
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()

    plt.yticks(data1['x_manual'], data1[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    ax.set_xlabel(xlabel)
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if hide_ticks == 'yes':
        ax.set_yticklabels([]) # hide y axis tick labels
    
    # Hard code x axis tick labels
    x_formatter = FixedFormatter(['0','0.5','1'])
    x_locator = FixedLocator([0,0.5,1])
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_locator(x_locator)
    
    # Add gridlines
    ax.grid(True, which="major")
    ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
    ax.minorticks_on()
    # Now hide the minor ticks (but leave the gridlines).
    ax.tick_params(which='minor', bottom=False, left=False)
    # Set how many minor gridlines to show between major gridlines.
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    
    return data1


# -----------------------------------------------------------------------------
# 2 SERIES bar plot for crude rates
def plot_bar_2series(dataset_1, dataset_2, x_fieldname, y_fieldname, xlims, ylims, width, height, y_pos_manual, color_list, fontsize, invert_axis, offset, scalar, outcome_var_categorical, outcome_var_dummy, hide_ticks, xlabel):
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    
    # Get unique dummy variable names and calculate counts and proportions
    # Dataset 1
    data1 = pd.DataFrame(model_results_combined_filter['Variable']).drop_duplicates()
    for var in data1['Variable']:
        data1['total_count_n'] = dataset_1.shape[0]
        data1.loc[(data1['Variable'] == var), 'group_count'] = dataset_1[~(dataset_1[outcome_var_categorical].isin(missing_data_values))][var].sum(axis = 0)
        data1.loc[(data1['Variable'] == var), 'outcome_count'] = dataset_1[(dataset_1[outcome_var_dummy] == 1)][var].sum(axis = 0)
        data1['group_prop'] = data1['group_count'] / data1['total_count_n']
        data1['outcome_prop'] = data1['outcome_count'] / data1['group_count']
        data1['y_pos_manual'] = data1['Variable'].map(Biobank_Recovery_Codebook.dictionary['y_pos_manual_with_reference'])
    data1['Variable_tidy'] = data1['Variable'].map(Biobank_Recovery_Codebook.dictionary['variable_tidy_V2'])
    data1 = data1[~(data1['y_pos_manual'].isnull())] # drop rows without numbers - not for plotting
    
    # Drop variables which are continuous - not for plotting - find by having 'unit:' in Variable_tidy string
    data1 = data1[~(data1['Variable_tidy'].str.contains('unit:'))] 
    
    # Dataset 2
    data2 = pd.DataFrame(model_results_combined_filter['Variable']).drop_duplicates()
    for var in data2['Variable']:
        data2['total_count_n'] = dataset_2.shape[0]
        data2.loc[(data2['Variable'] == var), 'group_count'] = dataset_2[~(dataset_2[outcome_var_categorical].isin(missing_data_values))][var].sum(axis = 0)
        data2.loc[(data2['Variable'] == var), 'outcome_count'] = dataset_2[(dataset_2[outcome_var_dummy] == 1)][var].sum(axis = 0)
        data2['group_prop'] = data2['group_count'] / data2['total_count_n']
        data2['outcome_prop'] = data2['outcome_count'] / data2['group_count']
        data2['y_pos_manual'] = data2['Variable'].map(Biobank_Recovery_Codebook.dictionary['y_pos_manual_with_reference'])
    data2['Variable_tidy'] = data2['Variable'].map(Biobank_Recovery_Codebook.dictionary['variable_tidy_V2'])
    data2 = data2[~(data2['y_pos_manual'].isnull())] # drop rows without numbers - not for plotting
    
    # Drop variables which are continuous - not for plotting - find by having 'unit:' in Variable_tidy string
    data2 = data2[~(data2['Variable_tidy'].str.contains('unit:'))] 
    
    if y_pos_manual == 'yes':
        data1['x_manual'] = (data1['y_pos_manual'])
        data2['x_manual'] = (data2['y_pos_manual'])
    else:
        data1['x_manual'] = (np.arange(len(data1[x_fieldname])))
        data2['x_manual'] = (np.arange(len(data2[x_fieldname])))
    
    # plot bars
    offset = offset
    fig, ax = plt.subplots(figsize=(width,height))
    # plt.figure(figsize=(width,height))
    ax.barh(data1['x_manual']-offset, data1[y_fieldname], height = offset*scalar, color = color_list[0], align='center')
    ax.barh(data2['x_manual']+offset, data2[y_fieldname], height = offset*scalar, color = color_list[1], align='center')
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()

    plt.yticks(data1['x_manual'], data1[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    ax.set_xlabel(xlabel)
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if hide_ticks == 'yes':
        ax.set_yticklabels([]) # hide y axis tick labels
    
    # Hard code x axis tick labels
    x_formatter = FixedFormatter(['0','0.5','1'])
    x_locator = FixedLocator([0,0.5,1])
    ax.xaxis.set_major_formatter(x_formatter)
    ax.xaxis.set_major_locator(x_locator)
    
    # Add gridlines
    ax.grid(True, which="major")
    ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
    ax.minorticks_on()
    # Now hide the minor ticks (but leave the gridlines).
    ax.tick_params(which='minor', bottom=False, left=False)
    # Set how many minor gridlines to show between major gridlines.
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    
    return data1, data2
    

# -----------------------------------------------------------------------------
# 3 SERIES scatter plot
def plot_OR_w_conf_int_3plots(data1, data2, data3, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, plot3_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, x_logscale, x_major_tick, x_minor_tick, poisson_reg):       
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['y_pos_manual']*scalar) + offset 
        data2['x_manual'] = (data2['y_pos_manual']*scalar)
        data3['x_manual'] = (data3['y_pos_manual']*scalar) - offset
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + offset 
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar)
        data3['x_manual'] = (np.arange(len(data3[x_fieldname]))*scalar) - offset
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(width,height))
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    # Plot 2
    ax = data2.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[1], s = 5, label = plot2_label, ax = ax)
    error_bar2 = ax.errorbar(y = data2['x_manual'], x = data2[y_fieldname], xerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    # Plot 3
    ax = data3.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "s", color = color_list[2], s = 6, label = plot3_label, ax = ax)
    error_bar3 = ax.errorbar(y = data3['x_manual'], x = data3[y_fieldname], xerr = np.array(data3[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[2])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.yticks(data3['x_manual'] + (offset), data3[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    if poisson_reg == 'yes':
        ax.set_xlabel('Relative risk ratio')
    else:
        ax.set_xlabel('Odds ratio')
        
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way
    
    if x_logscale == 'yes':
        ax.set_xscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twiny()
        secax.set_xscale('linear')
        secax.set_xlim(np.log(xlims[0]), np.log(xlims[1]))
        secax.grid(False, which="both", axis = 'x')
        secax.set_xlabel('Coefficient')
        secax.tick_params(axis='x', which='major', labelsize=10)
        secax.set_xlabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', labelsize=8.2)
        xlim_thresh = 3.5
        if xlims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '', '0.8', '',                                   
                                      '','','','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif xlims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '1.5','2', 
                                      '', '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.5, 2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        ax.xaxis.set_minor_formatter(x_formatter)
        ax.xaxis.set_minor_locator(x_locator)
                
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    
    if x_major_tick > 0:
        secax.xaxis.set_major_locator(MultipleLocator(x_major_tick))
    if x_minor_tick > 0:
        secax.xaxis.set_minor_locator(MultipleLocator(x_minor_tick))

    return data1, data2, data3


# -----------------------------------------------------------------------------
# 4 SERIES scatter plot
def plot_OR_w_conf_int_4plots(data1, data2, data3, data4, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, plot3_label, plot4_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, x_logscale, x_major_tick, x_minor_tick, poisson_reg):       
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['y_pos_manual']*scalar) + offset + (offset/2) 
        data2['x_manual'] = (data2['y_pos_manual']*scalar) + (offset/2)
        data3['x_manual'] = (data3['y_pos_manual']*scalar) - (offset/2)
        data4['x_manual'] = (data4['y_pos_manual']*scalar) - offset - (offset/2) 
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + offset + (offset/2) 
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar) + (offset/2)
        data3['x_manual'] = (np.arange(len(data3[x_fieldname]))*scalar) - (offset/2)
        data4['x_manual'] = (np.arange(len(data4[x_fieldname]))*scalar) - offset - (offset/2) 
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(width,height))
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    # Plot 2
    ax = data2.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[1], s = 5, label = plot2_label, ax = ax)
    error_bar2 = ax.errorbar(y = data2['x_manual'], x = data2[y_fieldname], xerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    # Plot 3
    ax = data3.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "s", color = color_list[2], s = 6, label = plot3_label, ax = ax)
    error_bar3 = ax.errorbar(y = data3['x_manual'], x = data3[y_fieldname], xerr = np.array(data3[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[2])
    # Plot 4
    ax = data4.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "^", color = color_list[3], s = 6, label = plot4_label, ax = ax)
    error_bar4 = ax.errorbar(y = data4['x_manual'], x = data4[y_fieldname], xerr = np.array(data4[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[3])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.yticks(data4['x_manual'] + offset + (offset/2), data4[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    if poisson_reg == 'yes':
        ax.set_xlabel('Relative risk ratio')
    else:
        ax.set_xlabel('Odds ratio')
        
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way
    
    if x_logscale == 'yes':
        ax.set_xscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twiny()
        secax.set_xscale('linear')
        secax.set_xlim(np.log(xlims[0]), np.log(xlims[1]))
        secax.grid(False, which="both", axis = 'x')
        secax.set_xlabel('Coefficient')
        secax.tick_params(axis='x', which='major', labelsize=10)
        secax.set_xlabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', labelsize=8.2)
        
        xlim_thresh = 3.5
        if xlims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '', '0.8', '',                                   
                                      '','','','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif xlims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '1.5','2', 
                                      '', '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.5, 2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        
        ax.xaxis.set_minor_formatter(x_formatter)
        ax.xaxis.set_minor_locator(x_locator)
        
                
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        
    if x_major_tick > 0:
        secax.xaxis.set_major_locator(MultipleLocator(x_major_tick))
    if x_minor_tick > 0:
        secax.xaxis.set_minor_locator(MultipleLocator(x_minor_tick))

    return data1, data2, data3, data4


# -----------------------------------------------------------------------------
# 5 SERIES scatter plot
def plot_OR_w_conf_int_5plots(data1, data2, data3, data4, data5, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, plot3_label, plot4_label, plot5_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, x_logscale, x_major_tick, x_minor_tick, poisson_reg):       
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['y_pos_manual']*scalar) + offset 
        data2['x_manual'] = (data2['y_pos_manual']*scalar) + (offset/2)
        data3['x_manual'] = (data3['y_pos_manual']*scalar)
        data4['x_manual'] = (data4['y_pos_manual']*scalar) - (offset/2)
        data5['x_manual'] = (data5['y_pos_manual']*scalar) - offset
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + offset 
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar) + (offset/2)
        data3['x_manual'] = (np.arange(len(data3[x_fieldname]))*scalar)
        data4['x_manual'] = (np.arange(len(data4[x_fieldname]))*scalar) - (offset/2)
        data5['x_manual'] = (np.arange(len(data5[x_fieldname]))*scalar) - offset
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(width,height))
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    # Plot 2
    ax = data2.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[1], s = 5, label = plot2_label, ax = ax)
    error_bar2 = ax.errorbar(y = data2['x_manual'], x = data2[y_fieldname], xerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    # Plot 3
    ax = data3.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "s", color = color_list[2], s = 6, label = plot3_label, ax = ax)
    error_bar3 = ax.errorbar(y = data3['x_manual'], x = data3[y_fieldname], xerr = np.array(data3[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[2])
    # Plot 4
    ax = data4.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "^", color = color_list[3], s = 6, label = plot4_label, ax = ax)
    error_bar4 = ax.errorbar(y = data4['x_manual'], x = data4[y_fieldname], xerr = np.array(data4[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[3])
    # Plot 5
    ax = data5.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "v", color = color_list[4], s = 6, label = plot5_label, ax = ax)
    error_bar5 = ax.errorbar(y = data5['x_manual'], x = data5[y_fieldname], xerr = np.array(data5[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[4])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.yticks(data5['x_manual'] + (offset), data5[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    if poisson_reg == 'yes':
        ax.set_xlabel('Relative risk ratio')
    else:
        ax.set_xlabel('Odds ratio')
        
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way
    
    if x_logscale == 'yes':
        ax.set_xscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twiny()
        secax.set_xscale('linear')
        secax.set_xlim(np.log(xlims[0]), np.log(xlims[1]))
        secax.grid(False, which="both", axis = 'x')
        secax.set_xlabel('Coefficient')
        secax.tick_params(axis='x', which='major', labelsize=10)
        secax.set_xlabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', labelsize=8.2)
        xlim_thresh = 3.5
        if xlims[1] < xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '0.3', '0.4', '0.5', '0.6', '', '0.8', '',                                   
                                      '','','','','1.5', '', '', '', '',
                                      '2', '', '', '', '', '2.5', '', '', '', '', 
                                      '3','','','','','',                               
                                      # '', '1.25', '', '1.67',
                                      # '2', '2.5', '3.33', 
                                      '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                                      2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                                      3.0,3.1,3.2,3.3,3.4,3.5,
                                      # 1.11, 1.25, 1.429, 1.66666, 
                                      # 2, 2.5, 3.333, 
                                      4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        elif xlims[1] >= xlim_thresh:
            x_formatter = FixedFormatter([#
                                          '0.02', '', '0.04', '', '0.06', '', '', '',
                                          # '0.02', '', '0.04', '', '0.06', '', '', '',
                                      '0.2', '', '0.4', '', '0.6', '', '', '',                                   
                                      '1.5','2', 
                                      '', '4', '', '6', '', '', '',
                                      '20', '', '40', '', '60'])
            x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                      0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                      1.5, 2, 
                                      3, 4, 5, 6, 7, 8, 9,
                                      20, 30, 40, 50, 60])
        
        ax.xaxis.set_minor_formatter(x_formatter)
        ax.xaxis.set_minor_locator(x_locator)
                
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        
    if x_major_tick > 0:
        secax.xaxis.set_major_locator(MultipleLocator(x_major_tick))
    if x_minor_tick > 0:
        secax.xaxis.set_minor_locator(MultipleLocator(x_minor_tick))

    return data1, data2, data3, data4, data5



# -----------------------------------------------------------------------------
# 6 SERIES scatter plot
def plot_OR_w_conf_int_6plots(data1, data2, data3, data4, data5, data6, x_fieldname, y_fieldname, conf_int_fieldnames, plot1_label, plot2_label, plot3_label, plot4_label, plot5_label, plot6_label, xlims, ylims, titlelabel, width, height, offset, y_pos_manual, color_list, fontsize, legend_offset, invert_axis, x_logscale, poisson_reg):       
    plt.rcParams.update({'font.size': fontsize}) # increase font size
    
    # Alt - try using xticks to create offset https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    scalar = 1 # scalar factor to increase size of axis
    
    if invert_axis == 'yes':
        offset = -1*offset
    
    if y_pos_manual == 'yes':
        # introduce offset to be able to separate markers
        data1['x_manual'] = (data1['y_pos_manual']*scalar) + offset 
        data2['x_manual'] = (data2['y_pos_manual']*scalar) + (3*(offset/5))
        data3['x_manual'] = (data3['y_pos_manual']*scalar) + (offset/5)
        data4['x_manual'] = (data4['y_pos_manual']*scalar) - (offset/5)
        data5['x_manual'] = (data5['y_pos_manual']*scalar) - (3*(offset/5))
        data6['x_manual'] = (data6['y_pos_manual']*scalar) - offset
    else:
        # introduce offset to be able to separate markers
        data1['x_manual'] = (np.arange(len(data1[x_fieldname]))*scalar) + offset 
        data2['x_manual'] = (np.arange(len(data2[x_fieldname]))*scalar) + (3*(offset/5))
        data3['x_manual'] = (np.arange(len(data3[x_fieldname]))*scalar) + (offset/5)
        data4['x_manual'] = (np.arange(len(data4[x_fieldname]))*scalar) - (offset/5)
        data5['x_manual'] = (np.arange(len(data5[x_fieldname]))*scalar) - (3*(offset/5))
        data6['x_manual'] = (np.arange(len(data6[x_fieldname]))*scalar) - offset
    
    # Plot 1        
    ax = data1.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "o", color = color_list[0], s = 10, label = plot1_label, figsize=(width,height))
    error_bar1 = ax.errorbar(y = data1['x_manual'], x = data1[y_fieldname], xerr = np.array(data1[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[0])
    # Plot 2
    ax = data2.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "D", color = color_list[1], s = 5, label = plot2_label, ax = ax)
    error_bar2 = ax.errorbar(y = data2['x_manual'], x = data2[y_fieldname], xerr = np.array(data2[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[1])
    # Plot 3
    ax = data3.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "s", color = color_list[2], s = 6, label = plot3_label, ax = ax)
    error_bar3 = ax.errorbar(y = data3['x_manual'], x = data3[y_fieldname], xerr = np.array(data3[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[2])
    # Plot 4
    ax = data4.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "^", color = color_list[3], s = 6, label = plot4_label, ax = ax)
    error_bar4 = ax.errorbar(y = data4['x_manual'], x = data4[y_fieldname], xerr = np.array(data4[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[3])
    # Plot 5
    ax = data5.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = "v", color = color_list[4], s = 6, label = plot5_label, ax = ax)
    error_bar5 = ax.errorbar(y = data5['x_manual'], x = data5[y_fieldname], xerr = np.array(data5[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[4])
    # Plot 6
    ax = data6.plot(kind = 'scatter', y = 'x_manual', x = y_fieldname, marker = ">", color = color_list[5], s = 6, label = plot6_label, ax = ax)
    error_bar6 = ax.errorbar(y = data6['x_manual'], x = data6[y_fieldname], xerr = np.array(data6[conf_int_fieldnames].transpose()), alpha = 0.4, capsize = 3, label = None, fmt = 'none', color = color_list[5])
    
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
    
    plt.yticks(data6['x_manual'] + (offset), data6[x_fieldname]) # set labels manually 
    ax.yaxis.label.set_visible(False) # hide y axis title
    
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
    
    plt.axvline(x = 1.0, color = 'k', linestyle = 'dashed', linewidth = 0.8)#, label = 'OR = 1') # add line to show odds of 1
    if poisson_reg == 'yes':
        ax.set_xlabel('Relative risk ratio')
    else:
        ax.set_xlabel('Odds ratio')
        
    # ax.set_xscale('log')
    ax.set_title(titlelabel)
    # ax.grid(b = True) # add gridlines
    # ax.get_legend().remove() # remove legend
    ax.legend(bbox_to_anchor=(0.5, legend_offset), loc = 'lower center') # move legend out of the way
    
    if x_logscale == 'yes':
        ax.set_xscale('log')
        # Add major gridlines and format ticks
        ax.grid(True, which="major")       
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3g}')) # .1f
        # ax.xaxis.set_minor_formatter(NullFormatter())
        
        # Add secondary x axis at top to show raw coefficient before conversion to OR
        secax=ax.twiny()
        secax.set_xscale('linear')
        secax.set_xlim(np.log(xlims[0]), np.log(xlims[1]))
        secax.grid(False, which="both", axis = 'x')
        secax.set_xlabel('Coefficient')
        secax.tick_params(axis='x', which='major', labelsize=10)
        secax.set_xlabel('Coefficient')
        
        # Add sparse minor tick labels
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', labelsize=8.2)
        x_formatter = FixedFormatter([#
                                      '0.02', '', '0.04', '', '0.06', '', '', '',
                                      # '0.02', '', '0.04', '', '0.06', '', '', '',
                                   '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', 
                                   # '0.2', '', '0.4', '', '0.6', '', '', '', 
                                   '1.25', '1.5',
                                  '2', '', '4', '', '6', '', '', '',
                                  '20', '', '40', '', '60'])
        x_locator = FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                                  0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                  1.25, 1.5,
                                  2, 3, 4, 5, 6, 7, 8, 9,
                                  20, 30, 40, 50, 60])
        ax.xaxis.set_minor_formatter(x_formatter)
        ax.xaxis.set_minor_locator(x_locator)
                
    else:
        # Add gridlines
        ax.grid(True, which="major")
        ax.grid(True, which="minor", color='#EEEEEE', axis = 'x')
        ax.minorticks_on()
        # Now hide the minor ticks (but leave the gridlines).
        ax.tick_params(which='minor', bottom=False, left=False)
        # Set how many minor gridlines to show between major gridlines.
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    return data1, data2, data3, data4, data5, data6


#%% Load data
# -----------------------------------------------------------------------------
# Participant master list 
fname = r"CSSBiobankRegistered-MASTERlist.csv"
patientlist = pd.read_csv(fname)
patientlist['cssbiobank_id'] = patientlist['cssbiobank_id'].str.replace("c","C") # make all C capitalised in CSS biobank id


# -----------------------------------------------------------------------------
# Processed follow-up long covid questionnaire 2022 file
fname = r"CSSBiobank_LCQ2022_processed.csv"
data_LCQ2022 = pd.read_csv(fname)
data_LCQ2022['cssbiobank_id'] = data_LCQ2022['cssbiobank_id'].str.replace("c","C") # make all C capitalised in CSS biobank id
data_LCQ2022_cols = data_LCQ2022.columns.to_list()
data_LCQ2022_copy = data_LCQ2022.copy()

# Filter for active participants only
data_LCQ2022 = data_LCQ2022[(data_LCQ2022['cssbiobank_id'].isin(patientlist['cssbiobank_id']))]


# -----------------------------------------------------------------------------
# Covid symptom duration summary from analysis of ZOE CSS assessments
# Use dataset containing all participants at latest time point - end of April 2022, at which point tests no longer available for free so use this as cut-off
date_cutoff = '2022-04-28'
data_symptomduration = pd.read_csv(r"CSSBiobank_ZOESymptomDurationSummary_AssessmentCutoff_2022-04-28_TestCutoff_2022-04-28_FromSnapshot_20220530_StringencyLimit7and14days.csv") 
data_symptomduration_cols = data_symptomduration.columns.to_list()

# -----------------------------------------------------------------------------
# Combined dataset combining ZOE data, LCQ data, registration data, containing basic demographics and other covariates (conditions, vaccination status, Biobank long covid questionnaire, ZOE mental health and long covid questionnaires). Generated in Biobank_CombiningDatasets.py script 
data_combined = pd.read_csv(r"CSSBiobank_CombinedDataset_AllActive_WithoutNHSNo.csv")

# Filter for active participants only
data_combined = data_combined[(data_combined['cssbiobank_id'].isin(patientlist['cssbiobank_id']))]
cols_combined = data_combined.columns.to_list()


# -----------------------------------------------------------------------------
# Long table of mental health assessment scores collected in various biobank sub-studies and zoe mental health questionnaire. Generated in Biobank_CollatingMentalHealthAssessments.py script 
data_mentalhealthassessments = pd.read_csv(r"CSSBiobank_MentalHealthAssessments_FullLong.csv")

# Filter for active participants only
data_mentalhealthassessments = data_mentalhealthassessments[(data_mentalhealthassessments['cssbiobank_id'].isin(patientlist['cssbiobank_id']))]

cols_MH = data_mentalhealthassessments.columns.to_list()


# -----------------------------------------------------------------------------
# Load inverse probability of participation weights
data_IPW_LCQ2022 = pd.read_csv(r"CSSBiobank_LCQ2022_participation_IPW_stringencylimit14_CSSCutoff_2022-04-28.csv")
data_IPW_LCQ2021and2022 = pd.read_csv(r"CSSBiobank_LCQ2021and2022_participation_IPW_stringencylimit14_CSSCutoff_2022-04-28.csv")

# Filter for active participants only
data_IPW_LCQ2022 = data_IPW_LCQ2022[(data_IPW_LCQ2022['cssbiobank_id'].isin(patientlist['cssbiobank_id']))]
data_IPW_LCQ2022_cols = data_IPW_LCQ2022.columns.to_list()

data_IPW_LCQ2021and2022 = data_IPW_LCQ2021and2022[(data_IPW_LCQ2021and2022['cssbiobank_id'].isin(patientlist['cssbiobank_id']))]
data_IPW_LCQ2021and2022_cols = data_IPW_LCQ2021and2022.columns.to_list()


dictionary = {}

#%% Pre-processing questionnaire participation dataset
# -----------------------------------------------------------------------------
# Binary field aggregating partial and full
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus'].isin(['partial', 'full']))
                               , 'ResponseStatus_binary'] = 'fullORpartial'
data_LCQ2022['ResponseStatus_binary'] = data_LCQ2022['ResponseStatus_binary'].fillna(data_LCQ2022['ResponseStatus'])


# -----------------------------------------------------------------------------
# Assign 0 values to health care sought flag for all with symptomatic infection should have answered to separate from missing data
symptom_duration_list = ['1. Less than 2 weeks', '2. 2-4 weeks','3. 4-12 weeks','4. 3-6 months','5. 6-12 months','6. 12-18 months','7. 18-24 months','8. More than 24 months']
# GP
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus'].isin(['partial', 'full']))
                 & (data_LCQ2022['SingleLongest_AnyEvidence_Infection_MedicalHelp_GP'].isnull())
                 & (data_LCQ2022['SingleLongest_AnyEvidence_Infection_SymptomDuration'].isin(symptom_duration_list))
                               , 'SingleLongest_AnyEvidence_Infection_MedicalHelp_GP'] = 0
# NHS 111/24
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus'].isin(['partial', 'full']))
                 & (data_LCQ2022['SingleLongest_AnyEvidence_Infection_MedicalHelp_NHS111'].isnull())
                 & (data_LCQ2022['SingleLongest_AnyEvidence_Infection_SymptomDuration'].isin(symptom_duration_list))
                               , 'SingleLongest_AnyEvidence_Infection_MedicalHelp_NHS111'] = 0
# Urgent care
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus'].isin(['partial', 'full']))
                 & (data_LCQ2022['SingleLongest_AnyEvidence_Infection_MedicalHelp_UrgentCare'].isnull())
                 & (data_LCQ2022['SingleLongest_AnyEvidence_Infection_SymptomDuration'].isin(symptom_duration_list))
                               , 'SingleLongest_AnyEvidence_Infection_MedicalHelp_UrgentCare'] = 0
# Pharmacist
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus'].isin(['partial', 'full']))
                 & (data_LCQ2022['SingleLongest_AnyEvidence_Infection_MedicalHelp_Pharmacist'].isnull())
                 & (data_LCQ2022['SingleLongest_AnyEvidence_Infection_SymptomDuration'].isin(symptom_duration_list))
                               , 'SingleLongest_AnyEvidence_Infection_MedicalHelp_Pharmacist'] = 0

# Unable to get care
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus'].isin(['partial', 'full']))
                 & (data_LCQ2022['SingleLongest_AnyEvidence_Infection_MedicalHelp_Unable'].isnull())
                 & (data_LCQ2022['SingleLongest_AnyEvidence_Infection_SymptomDuration'].isin(symptom_duration_list))
                               , 'SingleLongest_AnyEvidence_Infection_MedicalHelp_Unable'] = 0

# Count
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus'].isin(['partial', 'full']))
                 & (data_LCQ2022['SingleLongest_AnyEvidence_Infection_MedicalHelp_Count'].isnull())
                 & (data_LCQ2022['SingleLongest_AnyEvidence_Infection_SymptomDuration'].isin(symptom_duration_list))
                               , 'SingleLongest_AnyEvidence_Infection_MedicalHelp_Count'] = 0

dictionary['count_cat4'] = {0:'0. None',
                            1:'1. One',
                            2:'2. Two',
                            3:'3. Three or more',
                            4:'3. Three or more',
                            5:'3. Three or more',
                            }
data_LCQ2022['SingleLongest_AnyEvidence_Infection_MedicalHelp_Count_cat4'] = data_LCQ2022['SingleLongest_AnyEvidence_Infection_MedicalHelp_Count'].map(dictionary['count_cat4'])

# -----------------------------------------------------------------------------
# Join unknown and prefer not to say for pandemic experiences
codebook['pandemic_experiences'] = {'Yes': 'Yes',
                               'No': 'No',
                               'Prefer not to answer': 'Unknown/Prefer not to answer',
                               np.nan: 'Unknown/Prefer not to answer',
                               
                               }
data_LCQ2022['CovidExperiences_A_LostJob'] = data_LCQ2022['CovidExperiences_A_LostJob'].map(codebook['pandemic_experiences'])
data_LCQ2022['CovidExperiences_B_Furlough'] = data_LCQ2022['CovidExperiences_B_Furlough'].map(codebook['pandemic_experiences'])
data_LCQ2022['CovidExperiences_C_UnableToPayBills'] = data_LCQ2022['CovidExperiences_C_UnableToPayBills'].map(codebook['pandemic_experiences'])
data_LCQ2022['CovidExperiences_D_LostAccomodation'] = data_LCQ2022['CovidExperiences_D_LostAccomodation'].map(codebook['pandemic_experiences'])
data_LCQ2022['CovidExperiences_E_UnableToAffordFood'] = data_LCQ2022['CovidExperiences_E_UnableToAffordFood'].map(codebook['pandemic_experiences'])
data_LCQ2022['CovidExperiences_F_UnableAccessMedication'] = data_LCQ2022['CovidExperiences_F_UnableAccessMedication'].map(codebook['pandemic_experiences'])
data_LCQ2022['CovidExperiences_G_UnableAccessCommunityCare'] = data_LCQ2022['CovidExperiences_G_UnableAccessCommunityCare'].map(codebook['pandemic_experiences'])
data_LCQ2022['CovidExperiences_H_UnableAccessSocialCare'] = data_LCQ2022['CovidExperiences_H_UnableAccessSocialCare'].map(codebook['pandemic_experiences'])
data_LCQ2022['CovidExperiences_I_UnableAccessHospitalAppt'] = data_LCQ2022['CovidExperiences_I_UnableAccessHospitalAppt'].map(codebook['pandemic_experiences'])
data_LCQ2022['CovidExperiences_J_UnableAccessMHAppt'] = data_LCQ2022['CovidExperiences_J_UnableAccessMHAppt'].map(codebook['pandemic_experiences'])
data_LCQ2022['CovidExperiences_K_CovidBereavement'] = data_LCQ2022['CovidExperiences_K_CovidBereavement'].map(codebook['pandemic_experiences'])


# -----------------------------------------------------------------------------
# Aggregate pandemic experiences counts
dictionary['count_cat2'] = {0:'0. None',
                            1:'1. One or more',
                            2:'1. One or more',
                            3:'1. One or more',
                            4:'1. One or more',
                            5:'1. One or more',
                            6:'1. One or more',
                            7:'1. One or more',
                            8:'1. One or more',
                            9:'1. One or more',
                            np.nan:'Unknown',
                            }

dictionary['count_cat5'] = {0:'0. None',
                            1:'1. One',
                            2:'2. Two',
                            3:'3. Three',
                            4:'4. Four or more',
                            5:'4. Four or more',
                            6:'4. Four or more',
                            7:'4. Four or more',
                            8:'4. Four or more',
                            9:'4. Four or more',
                            10:'4. Four or more',
                            11:'4. Four or more',
                            12:'4. Four or more',
                            13:'4. Four or more',
                            14:'4. Four or more',
                            15:'4. Four or more',
                            np.nan:'Unknown',
                            }
data_LCQ2022['CovidExperiences_All_Count_cat5'] = data_LCQ2022['CovidExperiences_All_Count'].map(dictionary['count_cat5'])

dictionary['count_cat3'] = {0:'0. None',
                            1:'1. One',
                            2:'2. Two or more',
                            3:'2. Two or more',
                            4:'2. Two or more',
                            }
data_LCQ2022['CovidExperiences_Economic_Count_cat3'] = data_LCQ2022['CovidExperiences_Economic_Count'].map(dictionary['count_cat3'])
data_LCQ2022['CovidExperiences_Economic_Count_cat2'] = data_LCQ2022['CovidExperiences_Economic_Count'].map(dictionary['count_cat2'])

dictionary['count_cat4'] = {0:'0. None',
                            1:'1. One',
                            2:'2. Two',
                            3:'3. Three or more',
                            4:'3. Three or more',
                            5:'3. Three or more',
                            np.nan:'Unknown',
                            }
data_LCQ2022['CovidExperiences_HealthSocialCare_Count_cat4'] = data_LCQ2022['CovidExperiences_HealthSocialCare_Count'].map(dictionary['count_cat4'])
data_LCQ2022['CovidExperiences_HealthSocialCare_Count_cat2'] = data_LCQ2022['CovidExperiences_HealthSocialCare_Count'].map(dictionary['count_cat2'])

# Aggregate count of new health conditions due to covid
data_LCQ2022['NewCondition_DueToCovid_Count_cat4'] = data_LCQ2022['NewCondition_DueToCovid_Count'].map(dictionary['count_cat4'])

# Aggregate count of services referred to and appointments received
data_LCQ2022['LongCovidReferral_Referral_Count_cat5'] = data_LCQ2022['LongCovidReferral_Referral_Count'].map(dictionary['count_cat5'])
data_LCQ2022['LongCovidReferral_Appointment_Count_cat5'] = data_LCQ2022['LongCovidReferral_Appointment_Count'].map(dictionary['count_cat5'])
data_LCQ2022['LongCovidReferral_Appointment_Count_cat2'] = data_LCQ2022['LongCovidReferral_Appointment_Count'].map(dictionary['count_cat2'])

# -----------------------------------------------------------------------------
# Change 'Prefer not to answer' to NaN unknown to avoid model failure, as only 1 'Prefer not to answer' response
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus_binary'] == 'fullORpartial') & (data_LCQ2022['FirstLanguage_cat3'].isin(['Prefer not to answer', np.nan, 'NaN'])), 'FirstLanguage_cat3'] = 'Prefer not to answer/not stated'


# -----------------------------------------------------------------------------
# Education - Combine missing data with prefer not to answer
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus_binary'] == 'fullORpartial') & (data_LCQ2022['EducationLevel_cat3'].isin(['0. Prefer not to answer', np.nan, 'NaN'])), 'EducationLevel_cat3'] = '0. Prefer not to answer/not stated'
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus_binary'] == 'fullORpartial') & (data_LCQ2022['EducationLevel_cat6'].isin(['0. Prefer not to answer', np.nan, 'NaN'])), 'EducationLevel_cat6'] = '0. Prefer not to answer/not stated'


# -----------------------------------------------------------------------------
# Group education into binary for intersectional strata - degree or not
codebook['edu_binary'] = {'2. University degree':'Degree',
                                '3. Postgraduate degree or higher':'Degree',
                                '1. Less than University degree or equivalent':'Nodegree',
                                '0. Prefer not to answer/not stated':'Nodegree',
                                }
data_LCQ2022['EducationLevel_cat2'] = data_LCQ2022['EducationLevel_cat3'].map(codebook['edu_binary'])

codebook['edu_cat3_forstrata'] = {'2. University degree':'Degree',
                                '3. Postgraduate degree or higher':'PostgradDegree',
                                '1. Less than University degree or equivalent':'Nodegree',
                                '0. Prefer not to answer/not stated':'Nodegree',
                                }
data_LCQ2022['EducationLevel_forstrata_cat3'] = data_LCQ2022['EducationLevel_cat3'].map(codebook['edu_cat3_forstrata'])


# -----------------------------------------------------------------------------
# Reallocate data where low numbers affect ability of model to converge
# Employment status - Add certain categories to 'Other'
data_LCQ2022.loc[(data_LCQ2022['EmploymentStatus'].isin(['In education at school/college/university, or in an apprenticeship',
                                                          'In unpaid/voluntary work','Looking after home or family'
                                                         ])), 'EmploymentStatus'] = 'Other'
# Fill missing data with 'Unknown'
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus_binary'] == 'fullORpartial') & (data_LCQ2022['EmploymentStatus'].isin([np.nan, 'NaN'])), 'EmploymentStatus'] = 'Unknown'



# Group low income groups to give similar sample size to other categories
codebook['household_income'] = {1: '1. Less than £20,000',
                                2: '1. Less than £20,000',
                                3: '1. Less than £20,000',
                                4: '4. £20,000-£29,999',
                                5: '5. £30,000-£39,999',
                                6: '6. £40,000-£49,999',
                                7: '7. £50,000-£74,999',
                                8: '8. £75,000-£99,999',
                                9: '9. £100,000 or more',
                                999901: '0. Prefer not to answer',
                                }
data_LCQ2022['HouseholdIncome_cat9'] = data_LCQ2022['c_loc_fu1_income'].map(codebook['household_income'])

# Income - Combine missing data with prefer not to answer
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus_binary'] == 'fullORpartial') & (data_LCQ2022['HouseholdIncome_cat9'].isin(['0. Prefer not to answer', np.nan, 'NaN'])), 'HouseholdIncome_cat9'] = '0. Prefer not to answer/not stated'
data_LCQ2022.loc[(data_LCQ2022['ResponseStatus_binary'] == 'fullORpartial') & (data_LCQ2022['HouseholdIncome_cat5'].isin(['0. Prefer not to answer', np.nan, 'NaN'])), 'HouseholdIncome_cat5'] = '0. Prefer not to answer/not stated'



# -----------------------------------------------------------------------------
# Aggregate C2 symptom count - symptoms asked to not recovered
dictionary['count_cat_symptomcount'] = {0:'0. None',
                            1:'1. One',
                            2:'2. Two',
                            3:'3. Three',
                            4:'4. 4-5',
                            5:'4. 4-5',
                            6:'5. 6-7',
                            7:'5. 6-7',
                            8:'6. 8-9',
                            9:'6. 8-9',
                            10:'7. 10-14',
                            11:'7. 10-14',
                            12:'7. 10-14',
                            13:'7. 10-14',
                            14:'7. 10-14',
                            15:'7. 10-14',
                            }
data_LCQ2022['NotRecovered_Symptoms_Count_cat'] = data_LCQ2022['NotRecovered_Symptoms_Count'].map(dictionary['count_cat_symptomcount'])

# -----------------------------------------------------------------------------
# Convert C2 symptoms to string to use as categorical
symptom_col_list = ['NotRecovered_Symptoms_Breathing',
                  'NotRecovered_Symptoms_AlteredTasteSmell',
                  'NotRecovered_Symptoms_Thinking',
                  'NotRecovered_Symptoms_Heart',
                  'NotRecovered_Symptoms_LightHeaded',
                  'NotRecovered_Symptoms_Abdominal',
                  'NotRecovered_Symptoms_MuscleInclFatigue',
                  'NotRecovered_Symptoms_TinglingPain',
                  'NotRecovered_Symptoms_Mood',
                  'NotRecovered_Symptoms_Sleep',
                  'NotRecovered_Symptoms_SkinRashes',
                  'NotRecovered_Symptoms_BoneJointPain',
                  'NotRecovered_Symptoms_Headaches',
                  'NotRecovered_Symptoms_Infections',
                  'NotRecovered_Symptoms_Other']

for col in symptom_col_list:
    data_LCQ2022[col] = data_LCQ2022[col].astype(str)
    

# Rename ShieldingStatus due to overlap with ShieldingStatus_date columns
data_LCQ2022 = data_LCQ2022.rename(columns = {'ShieldingStatus':'ShieldingStatusOverall'})
    
#%% Pre-processing combined dataset
# -----------------------------------------------------------------------------
# Response status for LCQ 2021
data_combined.loc[(data_combined['Biobank_LCQ_Progress'] < 100)
                               , 'ResponseStatusLCQ2021'] = 'partial'
data_combined.loc[(data_combined['Biobank_LCQ_Progress'] == 100)
                               , 'ResponseStatusLCQ2021'] = 'full'
data_combined['ResponseStatusLCQ2021'] = data_combined['ResponseStatusLCQ2021'].fillna('invited')

# Binary field aggregating partial and full
data_combined.loc[(data_combined['ResponseStatusLCQ2021'].isin(['partial', 'full']))
                               , 'ResponseStatusLCQ2021_binary'] = 'fullORpartial'
data_combined['ResponseStatusLCQ2021_binary'] = data_combined['ResponseStatusLCQ2021_binary'].fillna(data_combined['ResponseStatusLCQ2021'])

# -----------------------------------------------------------------------------
# Make shielding flag numeric rather than string
dictionary['shielding'] = {'no':0,
                           'yes':1,
                           '0.1 Unknown - Answer not provided':np.nan}
data_combined['Biobank_LCQ_A2_ShieldingFlag'] = data_combined['Biobank_LCQ_A2_ShieldingFlag'].map(dictionary['shielding'])


data_combined.loc[(data_combined['Biobank_LCQ_A1_PrePandemicHealth'].isin([np.nan, 'NaN', '0.1 Unknown - Answer not provided'])), 'Biobank_LCQ_A1_PrePandemicHealth'] = 'Unknown'


# -----------------------------------------------------------------------------
# Aggregate IMD decile into 3 groups
dictionary['imd_cat3'] = {1:'1. Decile 1-3',
                          2:'1. Decile 1-3',
                          3:'1. Decile 1-3',
                          4:'2. Decile 4-7',
                          5:'2. Decile 4-7',
                          6:'2. Decile 4-7',
                          7:'2. Decile 4-7',
                          8:'3. Decile 8-10',
                          9:'3. Decile 8-10',
                          10:'3. Decile 8-10',
                          }
data_combined['Combined_IMD_cat3'] = data_combined['Combined_IMD_Decile'].map(dictionary['imd_cat3'])

# -----------------------------------------------------------------------------
# Convert IMD quintile and decile to use as categorical
data_combined['Combined_IMD_Quintile'] = data_combined['Combined_IMD_Quintile'].astype(str)
data_combined['Combined_IMD_Decile'] = data_combined['Combined_IMD_Decile'].astype(str)


# -----------------------------------------------------------------------------
# Aggregate physical condition count into 3 groups
dictionary['physical_condition_count'] = {0:'0 conditions',
                                          1:'1 condition',
                                          2:'2+ conditions',
                                          3:'2+ conditions',
                                          4:'2+ conditions',
                                          5:'2+ conditions',
                                          6:'2+ conditions',
                                          }
data_combined['ZOE_conditions_condition_count_cat3'] = data_combined['ZOE_conditions_condition_count_max'].map(dictionary['physical_condition_count'])


# -----------------------------------------------------------------------------
# Aggregate mental health condition count into groups
dictionary['mentalhealth_condition_count'] = {0:'0 conditions',
                                              1:'1 condition',
                                              2:'2 conditions',
                                              3:'3+ conditions',
                                              4:'3+ conditions',
                                              5:'3+ conditions',
                                              6:'3+ conditions',
                                              7:'3+ conditions',
                                              8:'3+ conditions',
                                              9:'3+ conditions',
                                              np.nan: 'Unknown'
                                              }
data_combined['ZOE_mentalhealth_condition_cat4'] = data_combined['ZOE_mentalhealth_condition_count'].map(dictionary['mentalhealth_condition_count'])


# -----------------------------------------------------------------------------
# Combine 0-30 with 30-40 and 80+ with 70-80 due to low numbers
data_combined.loc[(data_combined['Combined_Age_2022_grouped_decades'].isin(['1: 0-30','2: 30-40'])), 'Combined_Age_2022_grouped_decades'] = '1&2: 0-40'
data_combined.loc[(data_combined['Combined_Age_2022_grouped_decades'].isin(['6: 70-80','7: 80+'])), 'Combined_Age_2022_grouped_decades'] = '6&7: 70+'


# -----------------------------------------------------------------------------
# Combine Northern Ireland with Scotland due to low numbers
data_combined.loc[(data_combined['Region'].isin(['Scotland', 'Northern Ireland'])), 'Region'] = 'Scotland & Northern Ireland'


# -----------------------------------------------------------------------------
# Aggregate LCQ 2021 B14 symptom count - symptoms asked to all 
# Q.B.14 Did you have any of the following problems 12 weeks (or more) after first catching COVID-19?  Please only consider symptoms that are not explained by another reason. Tick all that apply.
dictionary['count_cat_symptomcount'] = {0:'0. None',
                            1:'1. One',
                            2:'2. Two',
                            3:'3. Three',
                            4:'4. 4-5',
                            5:'4. 4-5',
                            6:'5. 6-7',
                            7:'5. 6-7',
                            8:'6. 8-9',
                            9:'6. 8-9',
                            10:'7. 10-14',
                            11:'7. 10-14',
                            12:'7. 10-14',
                            13:'7. 10-14',
                            14:'7. 10-14',
                            15:'7. 10-14',
                            }
data_combined['Biobank_LCQ_B14_LongTermSymptoms_Count_cat'] = data_combined['Biobank_LCQ_B14_LongTermSymptoms_Count'].map(dictionary['count_cat_symptomcount'])

# -----------------------------------------------------------------------------
# Convert symptoms to string to use as categorical
symptom_B14_list = ['Biobank_LCQ_B14_LongTermSymptoms_Breathing',
                'Biobank_LCQ_B14_LongTermSymptoms_AlteredTasteSmell',
                'Biobank_LCQ_B14_LongTermSymptoms_Thinking',
                'Biobank_LCQ_B14_LongTermSymptoms_Heart',
                'Biobank_LCQ_B14_LongTermSymptoms_LightHeaded',
                'Biobank_LCQ_B14_LongTermSymptoms_Abdominal',
                'Biobank_LCQ_B14_LongTermSymptoms_MuscleInclFatigue',
                'Biobank_LCQ_B14_LongTermSymptoms_TinglingPain',
                'Biobank_LCQ_B14_LongTermSymptoms_Mood',
                'Biobank_LCQ_B14_LongTermSymptoms_Sleep',
                'Biobank_LCQ_B14_LongTermSymptoms_SkinRashes',
                'Biobank_LCQ_B14_LongTermSymptoms_BoneJointPain',
                'Biobank_LCQ_B14_LongTermSymptoms_Headaches',
                ]

for col in symptom_B14_list:
    data_combined[col] = data_combined[col].astype(str)


                                                                                     
#%% Pre-processing mental health assessment summary dataset
# Filter for assessments recorded before invitation to LCQ 2022
invite_1_date = '2022-08-30'

data_mentalhealthassessments_filter_BeforeLCQ2022 = data_mentalhealthassessments[(data_mentalhealthassessments['ResponseDate'] < invite_1_date)].copy()

def group_mentalhealth_assessments(df_long_combined):
    # -----------------------------------------------------------------------------
    # Group to get count, median, mean, standard deviation, min, max and most recent score
    # Sort by id and date
    df_long_combined = df_long_combined.sort_values(by = ['cssbiobank_id','Assessment','ResponseDate']).reset_index(drop = True)
    
    # Score metrics
    df_grouped_score = df_long_combined.groupby(['cssbiobank_id','Assessment']).agg({'score':['count','min','max','median','mean','std','first','last'],
                                                                                     'ResponseDate':['min','max'],
                                                                                     })
    
    df_grouped_score.columns = ['_'.join(col).strip() for col in df_grouped_score.columns.values]
    df_grouped_score = df_grouped_score.reset_index()
    
    # Add category field for mean and median scores
    df_grouped_score.loc[(df_grouped_score['Assessment'].isin(['GAD2', 'PHQ2'])) &
                         (df_grouped_score['score_mean'] >= 0) & (df_grouped_score['score_mean'] < 3)
                             ,'score_mean_cat'] = '1. 0-3, below threshold'
    df_grouped_score.loc[(df_grouped_score['Assessment'].isin(['GAD2', 'PHQ2'])) &
                         (df_grouped_score['score_mean'] >= 3) & (df_grouped_score['score_mean'] <= 6)
                             ,'score_mean_cat'] = '2. 3-6, above threshold'
    
    df_grouped_score.loc[(df_grouped_score['Assessment'].isin(['PHQ4'])) &
                         (df_grouped_score['score_mean'] >= 0) & (df_grouped_score['score_mean'] < 3)
                             ,'score_mean_cat'] = '1. 0-3, below threshold'
    df_grouped_score.loc[(df_grouped_score['Assessment'].isin(['PHQ4'])) &
                         (df_grouped_score['score_mean'] >= 3) & (df_grouped_score['score_mean'] < 6)
                             ,'score_mean_cat'] = '2. 3-6, mild'
    df_grouped_score.loc[(df_grouped_score['Assessment'].isin(['PHQ4'])) &
                         (df_grouped_score['score_mean'] >= 6) & (df_grouped_score['score_mean'] < 9)
                             ,'score_mean_cat'] = '3. 6-9, moderate'
    df_grouped_score.loc[(df_grouped_score['Assessment'].isin(['PHQ4'])) &
                         (df_grouped_score['score_mean'] >= 9) & (df_grouped_score['score_mean'] <= 12)
                             ,'score_mean_cat'] = '4. 9-12, severe'
                         
                         
    df_grouped_score.loc[(df_grouped_score['Assessment'].isin(['GAD2', 'PHQ2'])) &
                         (df_grouped_score['score_median'] >= 0) & (df_grouped_score['score_median'] < 3)
                             ,'score_median_cat'] = '1. 0-3, below threshold'
    df_grouped_score.loc[(df_grouped_score['Assessment'].isin(['GAD2', 'PHQ2'])) &
                         (df_grouped_score['score_median'] >= 3) & (df_grouped_score['score_median'] <= 6)
                             ,'score_median_cat'] = '2. 3-6, above threshold'
    
    df_grouped_score.loc[(df_grouped_score['Assessment'].isin(['PHQ4'])) &
                         (df_grouped_score['score_median'] >= 0) & (df_grouped_score['score_median'] < 3)
                             ,'score_median_cat'] = '1. 0-3, below threshold'
    df_grouped_score.loc[(df_grouped_score['Assessment'].isin(['PHQ4'])) &
                         (df_grouped_score['score_median'] >= 3) & (df_grouped_score['score_median'] < 6)
                             ,'score_median_cat'] = '2. 3-6, mild'
    df_grouped_score.loc[(df_grouped_score['Assessment'].isin(['PHQ4'])) &
                         (df_grouped_score['score_median'] >= 6) & (df_grouped_score['score_median'] < 9)
                             ,'score_median_cat'] = '3. 6-9, moderate'
    df_grouped_score.loc[(df_grouped_score['Assessment'].isin(['PHQ4'])) &
                         (df_grouped_score['score_median'] >= 9) & (df_grouped_score['score_median'] <= 12)
                             ,'score_median_cat'] = '4. 9-12, severe'
    
    
    # Sort by id and score
    df_long_combined = df_long_combined.sort_values(by = ['cssbiobank_id','Assessment','score']).reset_index(drop = True)
    
    # Category metrics
    df_grouped_cat = df_long_combined.groupby(['cssbiobank_id','Assessment']).agg({'ResponseDate':['first','last'],
                                                                                   'cat2':['first','last'],
                                                                                   'cat4':['first','last'],
                                                                                   })
    
    df_grouped_cat.columns = ['_'.join(col).strip() for col in df_grouped_cat.columns.values]
    df_grouped_cat = df_grouped_cat.rename(columns = {'cat2_first':'cat2_score_min',
                                            'cat2_last':'cat2_score_max',
                                            'cat4_first':'cat4_score_min',
                                            'cat4_last':'cat4_score_max',
                                            'ResponseDate_first':'ResponseDate_score_min',
                                            'ResponseDate_last':'ResponseDate_score_max',
                                            })
    
    # -----------------------------------------------------------------------------
    # Pivot from long to wide
    df_wide_scores = df_long_combined.pivot(index = ['cssbiobank_id', 'Assessment'], columns = 'StudyName', values = ['ResponseDate', 'score'])
    df_wide_scores.columns = ['_'.join(col).strip() for col in df_wide_scores.columns.values]
    df_wide_scores = df_wide_scores.reset_index(drop = False)
    
    # -----------------------------------------------------------------------------
    # Join wide scores to grouped score stats
    df_merge = pd.merge(df_wide_scores, df_grouped_score, how = 'left', on = ['cssbiobank_id','Assessment'])
    # Join grouped category stats
    df_merge = pd.merge(df_merge, df_grouped_cat, how = 'left', on = ['cssbiobank_id','Assessment'])
    
    return df_merge


# Generate summary table
data_mentalhealthassessments_wide_BeforeLCQ2022 = group_mentalhealth_assessments(data_mentalhealthassessments_filter_BeforeLCQ2022)

# Filter for relevant columns
cols_MH_select = ['cssbiobank_id', 'Assessment', 'score_mean', 'score_std', 'score_mean_cat', 'score_median_cat']
data_mentalhealthassessments_wide_BeforeLCQ2022 = data_mentalhealthassessments_wide_BeforeLCQ2022[cols_MH_select].copy()

# Filter for assessment of interest - start with PHQ-4 as contains most info, covers both anxiety and depression sub-scales
assessment_choose = 'PHQ4'
data_mentalhealthassessments_wide_BeforeLCQ2022 = data_mentalhealthassessments_wide_BeforeLCQ2022[(data_mentalhealthassessments_wide_BeforeLCQ2022['Assessment'] == assessment_choose)]



#%% Pre-processing symptom duration summary data
# -----------------------------------------------------------------------------
# Combine antigen and antibody specific categories in baseline pre-infection symptoms field
# First convert NaN to string
data_symptomduration.loc[(data_symptomduration['Flag_BaselineSymptoms_stringencylimit'+stringency].isnull())
                         ,'Flag_BaselineSymptoms_stringencylimit'+stringency] = 'NaN'

# Aggregate categories
data_symptomduration.loc[(data_symptomduration['Flag_BaselineSymptoms_stringencylimit'+stringency].str.contains('0.'))
                         ,'Flag_BaselineSymptoms_stringencylimit'+stringency] = '0. Unknown'
data_symptomduration.loc[(data_symptomduration['Flag_BaselineSymptoms_stringencylimit'+stringency].isin(['1. No regular symptoms between 28 and 14 days before start of longest symptom period before antibody test','1. No regular symptoms between 28 and 14 days before antigen test']))
                         ,'Flag_BaselineSymptoms_stringencylimit'+stringency] = '1. No regular symptoms between -28 and -14 days'
data_symptomduration.loc[(data_symptomduration['Flag_BaselineSymptoms_stringencylimit'+stringency].isin(['2. 2+ symptoms reported once a week between 28 and 14 days before antigen test','2. 2+ symptoms reported once a week between 28 and 14 days before start of longest symptom period before antibody test']))
                         ,'Flag_BaselineSymptoms_stringencylimit'+stringency] = '2. 2+ symptoms reported once a week between -28 and -14 days'

# Convert NaN string back to np.nan
data_symptomduration.loc[(data_symptomduration['Flag_BaselineSymptoms_stringencylimit'+stringency] == 'NaN')
                         ,'Flag_BaselineSymptoms_stringencylimit'+stringency] = np.nan

# -----------------------------------------------------------------------------
# Set symptom count value as 0 for asymptomatic
data_symptomduration.loc[(data_symptomduration['symptom_count_max_stringencylimit'+stringency].isnull())
                         & (data_symptomduration['symptomduration_grouped1_stringencylimit'+stringency].str.contains('asymptomatic'))
                         ,'symptom_count_max_stringencylimit'+stringency] = 0

# -----------------------------------------------------------------------------
# Set hospital flag value as 0 for all individual with current value of NaN but with known symptom duration. To distinguish between no and unknown
data_symptomduration.loc[(data_symptomduration['Flag_InHospitalDuringSpell_stringencylimit'+stringency].isnull())
                         & ~(data_symptomduration['symptomduration_weeks_stringencylimit'+stringency].isnull())
                         ,'Flag_InHospitalDuringSpell_stringencylimit'+stringency] = 0

# -----------------------------------------------------------------------------
# Create separate flags for SARS-CoV-2 negative and positive groups
data_symptomduration.loc[(data_symptomduration['result_stringencylimit'+stringency] == 3)
                         ,'Flag_InHospitalDuringSpell_Negative_stringencylimit'+stringency] = data_symptomduration['Flag_InHospitalDuringSpell_stringencylimit'+stringency]
data_symptomduration.loc[(data_symptomduration['result_stringencylimit'+stringency] == 4)
                         ,'Flag_InHospitalDuringSpell_Positive_stringencylimit'+stringency] = data_symptomduration['Flag_InHospitalDuringSpell_stringencylimit'+stringency]


# -----------------------------------------------------------------------------
# Combine 0-2 and 2-4 weeks covid groups, and 4-8 and 8-12 weeks together, to be in line with NICE guidelines, and to increase power as 2-4 and 8-12 weeks groups too small to do meaningful analysis
data_symptomduration.loc[(data_symptomduration['symptomduration_grouped1_stringencylimit'+stringency].isin(['N2: Negative COVID-19, 0-2 weeks','N3: Negative COVID-19, 2-4 weeks']))
                         , 'symptomduration_grouped1_stringencylimit'+stringency] = 'N2&3: Negative COVID-19, 0-4 weeks'
data_symptomduration.loc[(data_symptomduration['symptomduration_grouped1_stringencylimit'+stringency].isin(['P2: Positive COVID-19, 0-2 weeks','P3: Positive COVID-19, 2-4 weeks']))
                         , 'symptomduration_grouped1_stringencylimit'+stringency] = 'P2&3: Positive COVID-19, 0-4 weeks'

data_symptomduration.loc[(data_symptomduration['symptomduration_grouped1_stringencylimit'+stringency].isin(['N4: Negative COVID-19, 4-8 weeks','N5: Negative COVID-19, 8-12 weeks']))
                         , 'symptomduration_grouped1_stringencylimit'+stringency] = 'N4&5: Negative COVID-19, 4-12 weeks'
data_symptomduration.loc[(data_symptomduration['symptomduration_grouped1_stringencylimit'+stringency].isin(['P4: Positive COVID-19, 4-8 weeks','P5: Positive COVID-19, 8-12 weeks']))
                         , 'symptomduration_grouped1_stringencylimit'+stringency] = 'P4&5: Positive COVID-19, 4-12 weeks'

# -----------------------------------------------------------------------------
# Create grouped fields containing symptom duration only, not test result + symptom duration
data_symptomduration.loc[(data_symptomduration['symptomduration_grouped1_stringencylimit'+stringency].str.contains('0:'))
                         , 'symptomduration_only_grouped_stringencylimit'+stringency] = '0: Unknown'
data_symptomduration.loc[(data_symptomduration['symptomduration_grouped1_stringencylimit'+stringency].str.contains('1:'))
                         , 'symptomduration_only_grouped_stringencylimit'+stringency] = '1: Asymptomatic'
data_symptomduration.loc[(data_symptomduration['symptomduration_grouped1_stringencylimit'+stringency].str.contains('2&3:'))
                         , 'symptomduration_only_grouped_stringencylimit'+stringency] = '2&3: 0-4 weeks'
data_symptomduration.loc[(data_symptomduration['symptomduration_grouped1_stringencylimit'+stringency].str.contains('4&5:'))
                         , 'symptomduration_only_grouped_stringencylimit'+stringency] = '4&5: 4-8 weeks'
data_symptomduration.loc[(data_symptomduration['symptomduration_grouped1_stringencylimit'+stringency].str.contains('6:'))
                         , 'symptomduration_only_grouped_stringencylimit'+stringency] = '6: 12+ weeks'   

# -----------------------------------------------------------------------------
# Aggregate symptom count into 3 groups
data_symptomduration.loc[(data_symptomduration['symptom_count_max_stringencylimit'+stringency] == 0)
                          , 'symptom_count_max_cat4_stringencylimit'+stringency] = '1. 0 symptoms'
data_symptomduration.loc[(data_symptomduration['symptom_count_max_stringencylimit'+stringency] >= 1)
                         & (data_symptomduration['symptom_count_max_stringencylimit'+stringency] < 5)
                          , 'symptom_count_max_cat4_stringencylimit'+stringency] = '2. 1-4 symptoms'
data_symptomduration.loc[(data_symptomduration['symptom_count_max_stringencylimit'+stringency] >= 5)
                         & (data_symptomduration['symptom_count_max_stringencylimit'+stringency] < 10)
                          , 'symptom_count_max_cat4_stringencylimit'+stringency] = '3. 5-9 symptoms'                   
data_symptomduration.loc[(data_symptomduration['symptom_count_max_stringencylimit'+stringency] >= 10)
                          , 'symptom_count_max_cat4_stringencylimit'+stringency] = '4. 10+ symptoms'  

# -----------------------------------------------------------------------------
# Aggregate symptom duration into groups which match retrospective logging
# Add NaN to Unknown
data_symptomduration.loc[(data_symptomduration['symptomduration_only_grouped_stringencylimit'+stringency] == '0: Unknown')
                         | (data_symptomduration['symptomduration_only_grouped_stringencylimit'+stringency].isnull())
                          , 'symptomduration_only_grouped_stringencylimit'+stringency] = '0: Unknown'

# Split 0-4 weeks into finer groups
data_symptomduration.loc[(data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] > 0)
                         & (data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] < 2)
                          , 'symptomduration_only_grouped_stringencylimit'+stringency] = '2: 0-2 weeks'
data_symptomduration.loc[(data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] >= 2)
                         & (data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] < 4)
                          , 'symptomduration_only_grouped_stringencylimit'+stringency] = '3: 2-4 weeks'

# Split 12+ into finer groups
data_symptomduration.loc[(data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] >= 12)
                         & (data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] < 26)
                          , 'symptomduration_only_grouped_stringencylimit'+stringency] = '6a: 12-26 weeks'
data_symptomduration.loc[(data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] >= 26)
                         & (data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] < 52)
                          , 'symptomduration_only_grouped_stringencylimit'+stringency] = '6b: 26-52 weeks'
data_symptomduration.loc[(data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] >= 52)
                         & (data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] < 78)
                          , 'symptomduration_only_grouped_stringencylimit'+stringency] = '6c: 52-78 weeks'
data_symptomduration.loc[(data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] >= 78)
                         & (data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] < 104)
                          , 'symptomduration_only_grouped_stringencylimit'+stringency] = '6d: 78-104 weeks'
data_symptomduration.loc[(data_symptomduration['symptomduration_weeks_stringencylimit'+stringency] >= 104)
                          , 'symptomduration_only_grouped_stringencylimit'+stringency] = '6e: 104+ weeks'
 



#%% Join datasets together to create master table
# -----------------------------------------------------------------------------
# Create master table
data_master = data_LCQ2022.copy()

# -----------------------------------------------------------------------------
# Add mental health assessment data to master data table
# Assessment summary before round 1
data_master = pd.merge(data_LCQ2022, data_mentalhealthassessments_wide_BeforeLCQ2022.add_prefix('MH_BeforeLCQ2022_'), how = 'left', left_on = ['cssbiobank_id'], right_on = ['MH_BeforeLCQ2022_cssbiobank_id'])
data_master = data_master.drop(columns = ['MH_BeforeLCQ2022_cssbiobank_id'])

# -----------------------------------------------------------------------------
# Select covariates from CSS symptom duration summary dataset
cols_covariates_symptomduration = data_symptomduration.columns.to_list()
# Add covariates
data_master = pd.merge(data_master, data_symptomduration[cols_covariates_symptomduration], how = 'left', left_on = 'cssbiobank_id', right_on = 'cssbiobank_id')

# -----------------------------------------------------------------------------
# Select covariates from combined datasets
cols_covariates_combined = ['cssbiobank_id', 
                            'InvitationCohort',
                   'Combined_Age_2022', 'Combined_Age_2022_grouped_decades', 
                   'BiobankRegistration_Gender', 'BiobankRegistration_Gender_Other', 'ZOE_demogs_sex',
                   'Combined_EthnicityCategory', 'Combined_Ethnicity_cat2',
                   'Combined_BMI_value', 'Combined_BMI_cat5', 'Combined_BMI_cat2',
                   'Combined_IMD_Decile', 'Combined_IMD_Quintile', 'Combined_IMD_cat3', 'Region', 'Country', 'RUC_Latest2023',
                   'ZOE_demogs_healthcare_professional',
                   'ZOE_conditions_condition_count_max', 'ZOE_conditions_condition_count_cat3',
                   'ZOE_mentalhealth_generalised_anxiety_disorder',
                   'ZOE_mentalhealth_depression',
                   'ZOE_mentalhealth_condition_count', 'ZOE_mentalhealth_condition_cat4',
                   'ZOE_mentalhealth_ever_diagnosed_with_mental_health_condition',
                   'ZOE_vaccination_date_earliest_valid_vaccination',
                   'ZOE_vaccination_date_latest_valid_entry',
                   'PRISMA7_score','PRISMA7_cat2',
                   'ZOE_conditions_needs_help', 'ZOE_conditions_help_available', 'ZOE_conditions_mobility_aid', 'ZOE_conditions_limited_activity', 'ZOE_conditions_housebound_problems',
                   'ResponseStatusLCQ2021_binary',
                   'ResponseStatusLCQ2021',
                   'Biobank_LCQ_ResponseDate',
                   'Biobank_LCQ_A1_PrePandemicHealth',
                   'Biobank_LCQ_A2_ShieldingFlag',
                   'Biobank_LCQ_B7_HospitalisedFlag',
                   'Biobank_LCQ_B2_Post_Covid_Diagnosis_Self',
                   'Biobank_LCQ_B2_Post_Covid_Diagnosis_Doctor',
                   'Biobank_LCQ_B10_Recovered',
                   'Biobank_LCQ_B11_SymptomDuration',
                   'Biobank_LCQ_B12_AffectedFunctioningDuration',
                   'Combined_QuestionnaireSymptomDuration',
                   'Combined_QuestionnaireAffectedFunctioning',
                   'AntibodyTestResults_Value_Antibody_N',
                   
                   'Biobank_LCQ_B14_LongTermSymptoms_Breathing',
                    'Biobank_LCQ_B14_LongTermSymptoms_AlteredTasteSmell',
                    'Biobank_LCQ_B14_LongTermSymptoms_Thinking',
                    'Biobank_LCQ_B14_LongTermSymptoms_Heart',
                    'Biobank_LCQ_B14_LongTermSymptoms_LightHeaded',
                    'Biobank_LCQ_B14_LongTermSymptoms_Abdominal',
                    'Biobank_LCQ_B14_LongTermSymptoms_MuscleInclFatigue',
                    'Biobank_LCQ_B14_LongTermSymptoms_TinglingPain',
                    'Biobank_LCQ_B14_LongTermSymptoms_Mood',
                    'Biobank_LCQ_B14_LongTermSymptoms_Sleep',
                    'Biobank_LCQ_B14_LongTermSymptoms_SkinRashes',
                    'Biobank_LCQ_B14_LongTermSymptoms_BoneJointPain',
                    'Biobank_LCQ_B14_LongTermSymptoms_Headaches',
                    'Biobank_LCQ_B14_LongTermSymptoms_Count_cat',
                    
                    'Biobank_LCQ_B14_LongTermSymptoms_Count',
                    
                    'Biobank_LCQ_B17_UsefulSupport_SelfOrganisedGroup', 'Biobank_LCQ_B17_UsefulSupport_LocalVolunteer', 'Biobank_LCQ_B17_UsefulSupport_Household', 'Biobank_LCQ_B17_UsefulSupport_Neighbours', 'Biobank_LCQ_B17_UsefulSupport_ReligiousGroup', 'Biobank_LCQ_B17_UsefulSupport_Charity', 'Biobank_LCQ_B17_UsefulSupport_Family', 'Biobank_LCQ_B17_UsefulSupport_Friends', 'Biobank_LCQ_B17_UsefulSupport_LocalCouncil', 'Biobank_LCQ_B17_UsefulSupport_GPorNHS', 'Biobank_LCQ_B17_UsefulSupport_NotSure', 'Biobank_LCQ_B17_UsefulSupport_Count',
                    
                    'Biobank_LCQ_E1_PrePandemicEmploymentStatus', 'Biobank_LCQ_E2_CurrentEmploymentStatus',
                    # 'NonResponses_PriorToLCQ2022_Count',
                   ]

# Add covariates
data_master = pd.merge(data_master, data_combined[cols_covariates_combined], how = 'left', on = 'cssbiobank_id')

# Add inverse probability of questionnaire participation weights
data_master = pd.merge(data_master, data_IPW_LCQ2022, how = 'left', on = 'cssbiobank_id')
data_master = pd.merge(data_master, data_IPW_LCQ2021and2022, how = 'left', on = 'cssbiobank_id')


#%% Final processing after combining datasets
# -----------------------------------------------------------------------------
# Rename pre-pandemic employment to avoid duplication when searching variable names
data_master = data_master.rename(columns = {'Biobank_LCQ_E1_PrePandemicEmploymentStatus':'Biobank_LCQ_E1_StatusPrePandemicEmployment',
                                            'Biobank_LCQ_E2_CurrentEmploymentStatus':'Biobank_LCQ_E2_EmploymentCurrent'})


# Back-fill missing pre-pandemic employment status with current employment status from LCQ 2022 where LCQ 2022 response is 'Employed', 'Self-employed' or 'Other' only. ASSUMES NO CHANGE OVER TIME. Checked that bivariate association with recovery is similar for those with pre-pandemic data and missing data. Other categories like 'Retired', Permanently sick..., and Unemployed are compromised as COVID-19 illness may have caused change in employment status into one of these categories. 
data_master.loc[(data_master['Biobank_LCQ_E1_StatusPrePandemicEmployment'].isnull())
                 & (data_master['EmploymentStatus'].isin(['Employed','Retired','Self-employed']))
                 , 'Biobank_LCQ_E1_StatusPrePandemicEmployment'] = data_master['EmploymentStatus']


# -----------------------------------------------------------------------------
# Reallocate data where low numbers affect ability of model to converge
# Employment status - Add certain categories to 'Other'
data_master.loc[(data_master['Biobank_LCQ_E1_StatusPrePandemicEmployment'].isin(['In education at school/college/university, or in an apprenticeship',
                                                          'In unpaid/voluntary work','Looking after home or family', 'Carer', 'Part-time employed and/or partial retirement',
                                                         ])), 'Biobank_LCQ_E1_StatusPrePandemicEmployment'] = 'Other'

# Include not in work due to medical reasons to permanently sick or disabled
data_master.loc[(data_master['Biobank_LCQ_E1_StatusPrePandemicEmployment'].isin(['Permanently sick or disabled', 'Not in work due to medical reasons'])), 'Biobank_LCQ_E1_StatusPrePandemicEmployment'] = 'Permanently sick or disabled'

# Replace missing data with 'Unknown'
data_master.loc[(data_master['ResponseStatus_binary'] == 'fullORpartial') & (data_master['Biobank_LCQ_E1_StatusPrePandemicEmployment'].isin([np.nan, 'NaN'])), 'Biobank_LCQ_E1_StatusPrePandemicEmployment'] = 'Unknown'


# -----------------------------------------------------------------------------
# Calculate days between LCQ 2021 response and date of single longest infection - will use to filter later
test = data_master[['SingleLongest_AnyEvidence_Infection_DateEarliest','Biobank_LCQ_ResponseDate']]

data_master['DaysBetween_LCQ2021_SingleLongestInfection'] = (pd.to_datetime(data_master['SingleLongest_AnyEvidence_Infection_DateEarliest'], errors='coerce', format='%Y-%m-%d').dt.date - pd.to_datetime(data_master['Biobank_LCQ_ResponseDate'], errors='coerce', format='%Y-%m-%d').dt.date).dt.days

# -----------------------------------------------------------------------------
# Calculate days between LCQ 2022 response and date of single longest infection - will use to filter later
data_master['DaysBetween_LCQ2022_SingleLongestInfection'] = (pd.to_datetime(data_master['SingleLongest_AnyEvidence_Infection_DateEarliest'], errors='coerce', format='%Y-%m-%d').dt.date - pd.to_datetime(data_master['ResponseDate'], errors='coerce', format='%Y-%m-%d').dt.date).dt.days

# Flag if less than 28/84 days since infection
data_master.loc[(data_master['DaysBetween_LCQ2022_SingleLongestInfection'] <= 0) & (data_master['DaysBetween_LCQ2022_SingleLongestInfection'] > -28),'TimeSinceInfectionFlag'] = '0. 0-27 days since infection'
data_master.loc[(data_master['DaysBetween_LCQ2022_SingleLongestInfection'] <= -28) & (data_master['DaysBetween_LCQ2022_SingleLongestInfection'] > -84),'TimeSinceInfectionFlag'] = '1. 28-83 days since infection'
data_master.loc[(data_master['DaysBetween_LCQ2022_SingleLongestInfection'] <= -84),'TimeSinceInfectionFlag'] = '2. 84+ days since infection'

test = data_master[['SingleLongest_AnyEvidence_Infection_DateEarliest','Biobank_LCQ_ResponseDate','DaysBetween_LCQ2021_SingleLongestInfection']]



# -----------------------------------------------------------------------------
# Create intersectional strata variable that combines multiple positions
# Sex + Education (binary) + IMD (quintile)
# Sex
data_master['strata_sex_edu2_IMD5'] = data_master['ZOE_demogs_sex'].astype(str) + '_'
# Education as 2-cat
data_master['strata_sex_edu2_IMD5'] += data_master['EducationLevel_cat2'].astype(str) + '_'
# IMD as 5-cat
data_master['strata_sex_edu2_IMD5'] += data_master['Combined_IMD_Quintile'].astype(str)




#%% PRE-PROCESSING FINISHED, ANALYSES BEGIN HERE
# -----------------------------------------------------------------------------

#%% Apply general exclusions for all analysis and create datasets
# -----------------------------------------------------------------------------
# list of missing data strings
missing_data_values = [np.nan, 'NaN','nan', '0.1 Unknown - Answer not provided'] 

# -----------------------------------------------------------------------------
# Specify variables to do exclusions based on 
# Exclude those without basic demogs - age, sex, ethnicity, deprivation (proxy for address)
# DO NOT exclude those without a known symptom duration from CSS app data (and so an unknown COVID-19 infection and symptom duration group) as has been done for other studies, as for this study we have infection status and symptom duration collected in the LCQ questionnaire 
cols_exclude_demogs = [
                        'Combined_Age_2022', 
                        'ZOE_demogs_sex', 
                        'Combined_EthnicityCategory', 
                        'Combined_IMD_Decile',
                       ]

# -----------------------------------------------------------------------------
### Sleep study Round 1 - All invited
data_invited = data_master.copy()
print('Long covid questionnaire 2022 - All invited')
print('Number of individuals before exclusions: ' + str(data_invited.shape[0]))

# Main exclusion criteria - exclude those not invited to Round 1
data_invited = data_invited[(data_invited['ResponseStatus_binary'].isin(['invited','fullORpartial']))].reset_index(drop = True)   
print('Number of individuals after excluding based on ' + 'ResponseStatus_binary' + ': ' + str(data_invited.shape[0]))

# Exclude based on other key variables
for col in cols_exclude_demogs:
    data_invited = data_invited[~(data_invited[col].isin(missing_data_values))].reset_index(drop = True)   
    print('Number of individuals after excluding based on ' + col + ': ' + str(data_invited.shape[0]))


#%% Create dummy variables from un-ordered categoricals
# -----------------------------------------------------------------------------
# List of categorical input variables to create dummy variables for
variable_list_categorical = [# Participation in questionnaire
                             'ResponseStatus',
                             'ResponseStatus_binary',
                             'InvitationCohort',
                             
                             # Socio-demographics
                             'Combined_Age_2022_grouped_decades',
                             'ZOE_demogs_sex', 
                             'Combined_EthnicityCategory', 
                             'Combined_Ethnicity_cat2',
                             'Region',
                             'Country',
                             'Combined_IMD_cat3',
                             'Combined_IMD_Quintile',
                             'Combined_IMD_Decile',
                             'RUC_Latest2023',
                             'ZOE_demogs_healthcare_professional',
                             
                             # General health and wellbeing
                             'Combined_BMI_cat5',
                             'PRISMA7_cat2',
                             'ZOE_mentalhealth_ever_diagnosed_with_mental_health_condition',
                             'MH_BeforeLCQ2022_score_mean_cat',
                             'ZOE_conditions_condition_count_cat3',
                             'ZOE_mentalhealth_condition_cat4',
                             
                             # Illness characteristics from CSS app
                             'symptomduration_only_grouped_stringencylimit'+stringency,
                             'symptomduration_grouped1_stringencylimit'+stringency,
                             'symptom_count_max_cat4_stringencylimit'+stringency,
                             'result_stringencylimit'+stringency,
                             'Flag_InHospitalDuringSpell_stringencylimit'+stringency,
                             'Flag_InHospitalDuringSpell_Negative_stringencylimit'+stringency,
                             'Flag_InHospitalDuringSpell_Positive_stringencylimit'+stringency,
                             'Flag_BaselineSymptoms_stringencylimit'+stringency,                          
                             
                             # LCQ 2021 questionnaire
                             'Biobank_LCQ_B10_Recovered',
                             'Biobank_LCQ_E1_StatusPrePandemicEmployment',
                             'Biobank_LCQ_A1_PrePandemicHealth',
                             
                             # LCQ 2022 questionnaire - socio-demographics
                             'EducationLevel_cat3', 'EducationLevel_cat6', 'EmploymentStatus', 
                             'HouseholdIncome_cat9', 'HouseholdIncome_cat5', 'HouseholdIncome_cat4', 'HouseholdIncome_cat3',
                             'FirstLanguage_cat3',
                             
                             # LCQ 2022 questionnaire - experiences during pandemic
                             'CovidExperiences_All_Count_cat5',
                             'CovidExperiences_Economic_Count_cat3',
                             'CovidExperiences_Economic_Count_cat2',
                             'CovidExperiences_HealthSocialCare_Count_cat4',
                             'NewCondition_DueToCovid_Count_cat4',
                             'CovidExperiences_Personal_Count',
                             'CovidExperiences_Housing_Count',
                             'CovidExperiences_Employment_Count',
                             'CovidExperiences_A_LostJob', 
                             'CovidExperiences_B_Furlough',
                             'CovidExperiences_C_UnableToPayBills', 
                             'CovidExperiences_D_LostAccomodation',
                             'CovidExperiences_E_UnableToAffordFood',
                             'CovidExperiences_F_UnableAccessMedication', 
                             'CovidExperiences_G_UnableAccessCommunityCare',
                             'CovidExperiences_H_UnableAccessSocialCare', 
                             'CovidExperiences_I_UnableAccessHospitalAppt',
                             'CovidExperiences_J_UnableAccessMHAppt', 
                             'CovidExperiences_K_CovidBereavement',
                             
                             
                             # LCQ 2022 questionnaire - general health
                             'GeneralHealth', 'ShieldingStatusOverall',
                             
                             # LCQ 2022 questionnaire - covid illness
                             'SingleLongest_AnyEvidence_Infection_Evidence', 'SingleLongest_AnyEvidence_Infection_SymptomDuration',
                             
                             
                             'SingleLongest_AnyEvidence_Infection_MedicalHelp_Unable',
                             'SingleLongest_AnyEvidence_Infection_MedicalHelp_GP',
                             'SingleLongest_AnyEvidence_Infection_MedicalHelp_NHS111',
                             'SingleLongest_AnyEvidence_Infection_MedicalHelp_UrgentCare',
                             'SingleLongest_AnyEvidence_Infection_MedicalHelp_Pharmacist',
                             'SingleLongest_AnyEvidence_Infection_MedicalHelp_Count_cat4',
                             
                             'SingleLongest_AnyEvidence_Infection_Period',
                             'FunctionalImpairmentDuration_grouped',
                             'CovidRecovery',
                             
                             # LCQ 2022 questionnaire - long covid health care
                             'LongCovidDiagnosis',
                             'LongCovidReferral', 
                             'LongCovidReferral_Referral_Count_cat5', 'LongCovidReferral_Appointment_Count_cat5', 'LongCovidReferral_Appointment_Count_cat2',
                             'LongCovidReferral_Specialist_type_Combined',
                             
                             # LCQ 2022 questionnaire - current symptoms
                             'q_chalderFatigue_cat2', 'q_PHQ4_cat4', 'q_WSAS_cat4',
                             
                             # LCQ 2022 - individual symptoms among those not recovered
                             'NotRecovered_Symptoms_Breathing',
                             'NotRecovered_Symptoms_AlteredTasteSmell',
                             'NotRecovered_Symptoms_Thinking',
                             'NotRecovered_Symptoms_Heart',
                             'NotRecovered_Symptoms_LightHeaded',
                             'NotRecovered_Symptoms_Abdominal',
                             'NotRecovered_Symptoms_MuscleInclFatigue',
                             'NotRecovered_Symptoms_TinglingPain',
                             'NotRecovered_Symptoms_Mood',
                             'NotRecovered_Symptoms_Sleep',
                             'NotRecovered_Symptoms_SkinRashes',
                             'NotRecovered_Symptoms_BoneJointPain',
                             'NotRecovered_Symptoms_Headaches',
                             'NotRecovered_Symptoms_Infections',
                             'NotRecovered_Symptoms_Other',
                             'NotRecovered_Symptoms_Count_cat',
                             
                             # LCQ 2021 - individual symptoms experiences at 12+ weeks
                             'Biobank_LCQ_B14_LongTermSymptoms_Breathing',
                            'Biobank_LCQ_B14_LongTermSymptoms_AlteredTasteSmell',
                            'Biobank_LCQ_B14_LongTermSymptoms_Thinking',
                            'Biobank_LCQ_B14_LongTermSymptoms_Heart',
                            'Biobank_LCQ_B14_LongTermSymptoms_LightHeaded',
                            'Biobank_LCQ_B14_LongTermSymptoms_Abdominal',
                            'Biobank_LCQ_B14_LongTermSymptoms_MuscleInclFatigue',
                            'Biobank_LCQ_B14_LongTermSymptoms_TinglingPain',
                            'Biobank_LCQ_B14_LongTermSymptoms_Mood',
                            'Biobank_LCQ_B14_LongTermSymptoms_Sleep',
                            'Biobank_LCQ_B14_LongTermSymptoms_SkinRashes',
                            'Biobank_LCQ_B14_LongTermSymptoms_BoneJointPain',
                            'Biobank_LCQ_B14_LongTermSymptoms_Headaches',
                            'Biobank_LCQ_B14_LongTermSymptoms_Count_cat',
                            
                            # LCQ 2022 - new conditions due to covid
                            'c_loc_fu1_hcond_hypt_cov', 'c_loc_fu1_hcond_chol_cov', 'c_loc_fu1_hcond_diab_cov', 'c_loc_fu1_hcond_card_cov', 'c_loc_fu1_hcond_strok_cov', 'c_loc_fu1_hcond_clot_cov', 'c_loc_fu1_hcond_neur_cov', 'c_loc_fu1_hcond_renal_cov', 'c_loc_fu1_hcond_liver_cov', 'c_loc_fu1_hcond_canc_cov', 'c_loc_fu1_hcond_lung_cov', 'c_loc_fu1_hcond_dig_cov', 'c_loc_fu1_hcond_hear_cov', 'c_loc_fu1_hcond_arth_cov', 'c_loc_fu1_hcond_pain_cov', 'c_loc_fu1_hcond_meno_cov', 'c_loc_fu1_hcond_thyr_cov', 'c_loc_fu1_hcond_vit_cov', 'c_loc_fu1_hcond_mind_cov', 'c_loc_fu1_hcond_shin_cov', 'c_loc_fu1_hcond_posv_cov', 'c_loc_fu1_hcond_pcs_cov', 'c_loc_fu1_hcond_oth_cov',
                             
                            
                         ]

# Add dummy variables to datasets
data_invited = categorical_to_dummy(data_invited, variable_list_categorical)
data_invited_col_list = data_invited.columns.to_list()




#%% Apply inclusion & exclusion criteria to get analysis samples 
# -----------------------------------------------------------------------------
# Sample 0: All participants (full or partial completion)
data_0_allparticipants = data_invited[(data_invited['ResponseStatus_binary'] == 'fullORpartial')
             ].copy().reset_index()
data_0_allparticipants_cols = data_0_allparticipants.columns.to_list()


# -----------------------------------------------------------------------------
# Sample 1a: All participants (full or partial completion) with SYMPTOMATIC (asymptomatic don't get asked follow-up questions) self-reported infections, no exclusions based on symptom duration or evidence level
duration_symptomatic_list = ['1. Less than 2 weeks','2. 2-4 weeks','3. 4-12 weeks','4. 3-6 months','5. 6-12 months','6. 12-18 months','7. 18-24 months','8. More than 24 months']
covid_recovery_list = ['No, I still have some or all my symptoms','Yes, I am back to normal']
minimum_timesinceinfection = 83
data_1a_anyevidence_symptomatic = data_invited[(data_invited['ResponseStatus_binary'] == 'fullORpartial')
             & (data_invited['CovidRecovery'].isin(covid_recovery_list))
             & ~(data_invited['SingleLongest_AnyEvidence_Infection_Evidence'].isin(missing_data_values))
             & (data_invited['SingleLongest_AnyEvidence_Infection_SymptomDuration'].isin(duration_symptomatic_list))
             & (data_invited['DaysBetween_LCQ2022_SingleLongestInfection'] < -minimum_timesinceinfection) # Filter out infections less than a week before LCQ response, so recent infection not affecting recovery question response
             ].copy().reset_index()
data_1a_anyevidence_symptomatic_cols = data_1a_anyevidence_symptomatic.columns.to_list()

data_1a_anyevidence_symptomatic[~(data_1a_anyevidence_symptomatic['CovidRecovery'].isin(missing_data_values))]['Combined_Age_2022'].describe()


# Add flag to indicate which questionnaire respondents were selected into analysis sample
data_0_allparticipants.loc[data_0_allparticipants['cssbiobank_id'].isin(data_1a_anyevidence_symptomatic['cssbiobank_id']), 'SelectionIntoAnalysisFlag'] = 1
data_0_allparticipants['SelectionIntoAnalysisFlag'] = data_0_allparticipants['SelectionIntoAnalysisFlag'].fillna(0)


# -----------------------------------------------------------------------------
# Sample 2c: All participants (full or partial completion) with self-reported infections, Self-identified long covid by Covid diagnosis: No but think had long covid or Yes, Evidence level: Any
long_covid_list = ['No, but I do believe I have or have had Long COVID','Yes']
data_2c_anyevidence_selfreportlongcovid = data_invited[(data_invited['ResponseStatus_binary'] == 'fullORpartial')
             & ~(data_invited['SingleLongest_AnyEvidence_Infection_Evidence'].isin(missing_data_values))
             & (data_invited['LongCovidDiagnosis'].isin(long_covid_list))
             & (data_invited['DaysBetween_LCQ2022_SingleLongestInfection'] < -minimum_timesinceinfection) # Filter out infections less than a week before LCQ response, so recent infection not affecting recovery question response
             ].copy().reset_index()
data_2c_anyevidence_selfreportlongcovid_cols = data_2c_anyevidence_selfreportlongcovid.columns.to_list()

# Add flag to indicate which questionnaire respondents were selected into analysis sample
data_0_allparticipants.loc[data_0_allparticipants['cssbiobank_id'].isin(data_2c_anyevidence_selfreportlongcovid['cssbiobank_id']), 'SelectionIntoAnalysis_SelfReportLongCOVID_Flag'] = 1
data_0_allparticipants['SelectionIntoAnalysis_SelfReportLongCOVID_Flag'] = data_0_allparticipants['SelectionIntoAnalysis_SelfReportLongCOVID_Flag'].fillna(0)



#%% Specify reference categories 
# -----------------------------------------------------------------------------
# List of dummy fields to drop from model, to use as reference category
cols_categorical_reference = [# Invitation/admin related
                            'InvitationCohort_1. October-November 2020 COVID-19 invitation',
                            
                            # Socio-demographics
                            'Combined_Age_2022_grouped_decades_4: 50-60', # REFERENCE CATEGORY - Modal
                            'ZOE_demogs_sex_Female', # REFERENCE CATEGORY - Modal
                            'Combined_EthnicityCategory_White', # Modal
                            'Combined_Ethnicity_cat2_White', # Modal
                            'Region_London', # REFERENCE CATEGORY - Modal
                            'Country_England', # REFERENCE CATEGORY - Modal
                            'Combined_IMD_cat3_3. Decile 8-10',
                            'Combined_IMD_Quintile_5.0',
                            'Combined_IMD_Decile_10.0',
                            'RUC_Latest2023_Urban',
                            'ZOE_demogs_healthcare_professional_no',
                            
                            # General health and wellbeing
                            'Combined_BMI_cat5_2: 18.5-25',
                            'PRISMA7_cat2_1. 0-2, below threshold',
                            'ZOE_mentalhealth_ever_diagnosed_with_mental_health_condition_NO',
                            'MH_BeforeLCQ2022_score_mean_cat_1. 0-3, below threshold',
                            'ZOE_conditions_condition_count_cat3_0 conditions',
                            'ZOE_mentalhealth_condition_cat4_0 conditions',
                            
                            # Illness characteristics from CSS app
                            'symptomduration_only_grouped_stringencylimit'+stringency+'_2: 0-2 weeks',
                            'symptomduration_grouped1_stringencylimit'+stringency+'_P6: Positive COVID-19, 12+ weeks',
                             
                            'symptom_count_max_cat4_stringencylimit'+stringency+'_1. 0 symptoms',
                            'result_stringencylimit'+stringency+'_3.0',
                            'Flag_InHospitalDuringSpell_stringencylimit'+stringency+'_0.0',
                            'Flag_InHospitalDuringSpell_Negative_stringencylimit'+stringency+'_0.0',
                            'Flag_InHospitalDuringSpell_Positive_stringencylimit'+stringency+'_0.0',
                            'Flag_BaselineSymptoms_stringencylimit'+stringency+'_1. No regular symptoms between -28 and -14 days',
                            
                            # LCQ 2021
                            'Biobank_LCQ_B10_Recovered_NaN',
                            'Biobank_LCQ_E1_StatusPrePandemicEmployment_Employed',
                            'Biobank_LCQ_A1_PrePandemicHealth_4. Very Good',
                            
                            # LCQ 2022 questionnaire - socio-demographics
                            'EducationLevel_cat3_2. University degree',
                            'EducationLevel_cat6_4. University degree',
                             'EmploymentStatus_Employed', 
                             'HouseholdIncome_cat9_6. £40,000-£49,999', # Modal and median for CSSB
                             'HouseholdIncome_cat5_3. £40,000-£49,999', # Based on median of £46.2k from Unequivalised gross income, UK, 2020 unequivalisedgrossincomedeciles2020 from ONS
                             'HouseholdIncome_cat4_3. £50,000-£74,999', # Based on median of £46.2k from Unequivalised gross income, UK, 2020 unequivalisedgrossincomedeciles2020 from ONS
                             'HouseholdIncome_cat3_2. £40,000-£74,999', # Based on median of £46.2k from Unequivalised gross income, UK, 2020 unequivalisedgrossincomedeciles2020 from ONS
                             
                             'FirstLanguage_cat3_English',
                             
                             
                             
                             # LCQ 2022 questionnaire - experiences during pandemic
                             'CovidExperiences_All_Count_cat5_0. None',
                             'CovidExperiences_Economic_Count_cat3_0. None', 
                             'CovidExperiences_Economic_Count_cat2_0. None', 
                             'CovidExperiences_HealthSocialCare_Count_cat4_0. None',
                             'NewCondition_DueToCovid_Count_cat4_0. None',
                             'CovidExperiences_Personal_Count_0.0',
                             'CovidExperiences_Housing_Count_0.0',
                             'CovidExperiences_Employment_Count_0.0',
                             
                             'CovidExperiences_A_LostJob_No', 
                             'CovidExperiences_B_Furlough_No',
                             'CovidExperiences_C_UnableToPayBills_No', 
                             'CovidExperiences_D_LostAccomodation_No',
                             'CovidExperiences_E_UnableToAffordFood_No',
                             'CovidExperiences_F_UnableAccessMedication_No',
                             'CovidExperiences_G_UnableAccessCommunityCare_No',
                             'CovidExperiences_H_UnableAccessSocialCare_No', 
                             'CovidExperiences_I_UnableAccessHospitalAppt_No',
                             'CovidExperiences_J_UnableAccessMHAppt_No', 
                             'CovidExperiences_K_CovidBereavement_No',
                             
                             # LCQ 2022 questionnaire - general health
                             'GeneralHealth_3. Good', 
                             'ShieldingStatusOverall_No',
                             
                             # LCQ 2022 questionnaire - covid illness 
                             'SingleLongest_AnyEvidence_Infection_SymptomDuration_1. Less than 2 weeks',
                             
                             'SingleLongest_AnyEvidence_Infection_MedicalHelp_Unable_0.0',
                             'SingleLongest_AnyEvidence_Infection_MedicalHelp_GP_0.0',
                             'SingleLongest_AnyEvidence_Infection_MedicalHelp_NHS111_0.0',
                             'SingleLongest_AnyEvidence_Infection_MedicalHelp_UrgentCare_0.0',
                             'SingleLongest_AnyEvidence_Infection_MedicalHelp_Pharmacist_0.0',
                             'SingleLongest_AnyEvidence_Infection_MedicalHelp_Count_cat4_0. None',
                             
                             'SingleLongest_AnyEvidence_Infection_Period_1. Before 2020-12-08 (pre-vaccination, wild-type dominant)',
                             'FunctionalImpairmentDuration_grouped_1. Less than 2 weeks',
                             'CovidRecovery_Yes, I am back to normal',
                             
                             # LCQ 2022 questionnaire - long covid health care
                             'LongCovidDiagnosis_No, but I do believe I have or have had Long COVID',
                             'LongCovidReferral_No', 
                             'LongCovidReferral_Referral_Count_cat5_0. None',
                             'LongCovidReferral_Appointment_Count_cat5_0. None',
                             'LongCovidReferral_Appointment_Count_cat2_0. None',
                             
                             # LCQ 2022 questionnaire - current symptoms
                             'q_chalderFatigue_cat2_1. 0-28, below threshold',
                             'q_PHQ4_cat4_1. 0-2, below threshold', 
                             'q_WSAS_cat4_1. 0-9, below threshold',
                             
                             # LCQ 2022 - individual symptoms among those not recovered
                             'NotRecovered_Symptoms_Breathing_0.0',
                             'NotRecovered_Symptoms_AlteredTasteSmell_0.0',
                             'NotRecovered_Symptoms_Thinking_0.0',
                             'NotRecovered_Symptoms_Heart_0.0',
                             'NotRecovered_Symptoms_LightHeaded_0.0',
                             'NotRecovered_Symptoms_Abdominal_0.0',
                             'NotRecovered_Symptoms_MuscleInclFatigue_0.0',
                             'NotRecovered_Symptoms_TinglingPain_0.0',
                             'NotRecovered_Symptoms_Mood_0.0',
                             'NotRecovered_Symptoms_Sleep_0.0',
                             'NotRecovered_Symptoms_SkinRashes_0.0',
                             'NotRecovered_Symptoms_BoneJointPain_0.0',
                             'NotRecovered_Symptoms_Headaches_0.0',
                             'NotRecovered_Symptoms_Infections_0.0',
                             'NotRecovered_Symptoms_Other_0.0',
                             'NotRecovered_Symptoms_Count_cat_0. None',
                             
                             
                             # LCQ 2021 - individual symptoms experiences at 12+ weeks
                             'Biobank_LCQ_B14_LongTermSymptoms_Breathing_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_AlteredTasteSmell_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_Thinking_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_Heart_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_LightHeaded_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_Abdominal_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_MuscleInclFatigue_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_TinglingPain_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_Mood_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_Sleep_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_SkinRashes_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_BoneJointPain_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_Headaches_0.0',
                            'Biobank_LCQ_B14_LongTermSymptoms_Count_cat_3. Three',
                            
                        
                            
                            
                                   ]



#%% Creating prediction model for selection into analysis set: Run forward sequential feature selection to identify most predictive set of features
do_IPW_sfs = 0
do_poisson_IPW = '' # don't do poisson regression for IPW - use logistic regression
if do_IPW_sfs == 1:
    # -----------------------------------------------------------------------------
    # Set dataset and outcome variable to test
    # 0 = Model name, 1 = dataset, 2 = full fieldname list, 3 = outcome variable
    # SelectionIntoAnalysisFlag, SelectionIntoAnalysis_SelfReportLongCOVID_Flag 
    outcome_var = 'SelectionIntoAnalysis_SelfReportLongCOVID_Flag'
    manual_sfs_run_list = [['Predicting selection into analysis sample', 
                            data_0_allparticipants, # data_1a_anyevidence_symptomatic data_2c_anyevidence_selfreportlongcovid
                            data_0_allparticipants_cols, # data_1a_anyevidence_symptomatic_cols
                        outcome_var,],
                       ]
    
    manual_sfs_result_list = []
    best_model_vars_list = []
    
    for n in range(0, len(manual_sfs_run_list), 1):
        model_name = manual_sfs_run_list[n][0]
        dataset = manual_sfs_run_list[n][1]
        full_fieldname_list = manual_sfs_run_list[n][2]
        
        # -----------------------------------------------------------------------------
        # Set outcome field
        outcome_var = manual_sfs_run_list[n][3]
        
        # -----------------------------------------------------------------------------
        # Set input fields
        # Include variables with complete or near complete data, and related to either main exposure (covid group) or outcome (participation) in proposed DAG digram
        # List of categorical variables (non-dummy fields)
        input_var_categorical = [# Socio-demographics
                                     'Combined_Age_2022_grouped_decades',
                                     'ZOE_demogs_sex',
                                     'Combined_EthnicityCategory',
                                     'EducationLevel_cat6',
                                     'FirstLanguage_cat3',
                                     
                                     # Pre-pandemic social determinants of health 
                                     'Biobank_LCQ_E1_StatusPrePandemicEmployment',
                                     'Region',
                                     'Combined_IMD_Decile',                                 
                                     'ZOE_demogs_healthcare_professional',
                                     
                                     # Pre-pandemic health and wellbeing
                                     'Combined_BMI_cat5',
                                     'ZOE_conditions_condition_count_cat3',
                                     'Biobank_LCQ_A1_PrePandemicHealth',
                                     # 'ShieldingStatusOverall',
                                     
                           ]
            
        # List of continuous variables
        input_var_continuous = [
                                # Pre-pandemic health and wellbeing
                                'PRISMA7_score',
                                # 'NonResponses_PriorToLCQ2022_Count',
                                ]
        
        # -----------------------------------------------------------------------------
        # Generate list of categorical input dummy variables, by identifying all dummy variables with fieldname starting with a value in input_var_categorical and deleting reference variable using input_var_categorical_reference list
        
        # Generate list of dummy fields for complete fields
        input_var_categorical_formodel = generate_dummy_list(original_fieldname_list = input_var_categorical, 
                                                         full_fieldname_list = full_fieldname_list, 
                                                         reference_fieldname_list = cols_categorical_reference,
                                                         delete_reference = 'yes')
        
        
        # -----------------------------------------------------------------------------
        ### Run various models to measure importance of features
        # Set input fields
        # Combine categorical and continuous
        input_var_all = input_var_categorical_formodel + input_var_continuous
               
        # Drop dummy columns where sum of column = 0 - i.e. no-one from particular group - can cause 'Singular matrix' error when running model
        empty_cols  = []
        input_var_categorical_formodel_copy = input_var_categorical_formodel.copy()
        for col in input_var_categorical_formodel_copy: 
            # print(col)
            if dataset[col].sum() < 1:
                print('remove empty dummy: ' + col)
                input_var_categorical_formodel.remove(col)
                empty_cols.append(col)
        
        # Drop dummy columns where no observations of outcome in group in column = 0 - i.e. no observations - can cause 'Singular matrix' error when running model
        input_var_categorical_formodel_copy = input_var_categorical_formodel.copy()
        for col in input_var_categorical_formodel_copy: 
            if dataset[(dataset[outcome_var] == 1)][col].sum() < 1:
                print('remove dummy with no observations of outcome: ' + col)
                input_var_categorical_formodel.remove(col)
                empty_cols.append(col)
    
        print('After dropping MISSING DATA and EMPTY dummy cols: ')
        print(input_var_categorical_formodel)
        
        # Set variables to go into model
        input_var_all = input_var_continuous + input_var_categorical_formodel
        
        
        # Generate outcome and input data
        y_data = dataset[outcome_var].reset_index(drop=True) # set output variable
        x_data = dataset[input_var_all].reset_index(drop=True) # create input variable tables for models
    
        # Split data into test and train set
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, stratify = y_data, test_size = 0.25, random_state = 0)
        x_train = x_train.reset_index(drop = True)
        y_train = y_train.reset_index(drop = True)
        x_test = x_test.reset_index(drop = True)
        y_test = y_test.reset_index(drop = True)
        
        # # automatically drop columns where sum is below a threshold value - this means that there are very few observation that are = 1 and variable therefore not useful for model
        # x_data_sum_before = x_data.sum()
          
        # empty_cols  = []
        # for col in x_train: 
        #     if x_train[col].sum() <= 1:
        #         x_train = x_train.drop(columns = col)
        #         x_test = x_test.drop(columns = col)
        #         x_data = x_data.drop(columns = col)
        #         print('drop empty column: ' + col)
        #         empty_cols.append(col)
                
        # x_data_sum_after = x_data.sum().reset_index()
        # x_train_sum_after = x_train.sum().reset_index()
    
        # -----------------------------------------------------------------------------
        # Calculate correlation between variables to help feature selection 
        x_data_correlation = x_data.corr()
        # Flatten and filter to identify correlations larger than a threshold and decide which features to eliminate
        # Flatten dataframe
        x_data_correlation_flat = x_data_correlation.stack().reset_index()
        # Identify values above threshold 
        corr_thresh = 0.2
        x_data_correlation_above_thresh = x_data_correlation_flat[(x_data_correlation_flat[0].abs() >= corr_thresh)
                                                                  & (x_data_correlation_flat[0].abs() < 1)]
        
        # Create heatmap 
        # x_data_correlation_flat.columns = ['x', 'y', 'value']
        # af.heatmap(x=x_data_correlation_flat['x'], y=x_data_correlation_flat['y'], size=x_data_correlation_flat['value'].abs())
    
        
        # -----------------------------------------------------------------------------
        # Manual forward Sequential Feature Selection
        model_select = 'logreg' # rf, logreg, extratrees, boosting
        # List of potential variables
        var_list = input_var_categorical + input_var_continuous
        var_list_len = len(var_list)
        # Start with empty list of variables to add selected to
        selected_vars = []
        auc_max_list = []
        best_var_list = []
        for n in range (0,var_list_len,1):
            model_auc_list = []
            for var in var_list:
                # if variable categorical, pull out dummy variables for model
                if var in input_var_categorical:
                    # Generate list of dummy fields for complete fields
                    var_formodel = generate_dummy_list(original_fieldname_list = [var], 
                                                    full_fieldname_list = full_fieldname_list, 
                                                    reference_fieldname_list = cols_categorical_reference,
                                                    delete_reference = 'yes')
                else:
                    var_formodel = [var]
                
                vars_input = selected_vars + var_formodel
                
                # Drop any empty columns identified earlier
                for empty_col in empty_cols:
                    if empty_col in vars_input:
                        print('remove empty column: ' + empty_col + ' from input vars')
                        vars_input.remove(empty_col)
                
                if model_select in ['rf', 'boosting', 'extratrees']:
                    # Filter test and train data for selected variables
                    x_train_sfs = x_train[vars_input]
                    x_test_sfs = x_test[vars_input]
                    if model_select == 'rf':
                        model = RandomForestClassifier()
                    elif model_select == 'boosting':
                        model = GradientBoostingClassifier()
                    elif model_select == 'extratrees':
                        model = ExtraTreesClassifier()
                    model_fit = model.fit(x_train_sfs, y_train) # fit model with training data
                    model_y_pred = model.predict(x_test_sfs) # generate predictions for test data
                    model_auc = metrics.roc_auc_score(y_test, model_y_pred)
                
                elif model_select == 'logreg':
                    # Do logistic regression (stats models) of control + test variables
                    x_data_sfs = x_data[vars_input].copy()
                    sm_summary, model_fit, model_auc, model_explained_variance, model_r2 = sm_logreg_simple_HC3(x_data = x_data_sfs, y_data = y_data, 
                                                         CI_alpha = 0.05, do_robust_se = 'HC3',
                                                         use_weights = 'NA', weight_data = '',
                                                         do_poisson = do_poisson_IPW)
        
                print(str(vars_input)+': AUC: '+str(model_auc))
                # Save to list
                model_auc_list.append(model_auc)
            
            # Create dataframe
            auc_df = pd.DataFrame({'var':var_list,
                          'AUC':model_auc_list})
            
            # Identify variable with maximum AUC
            auc_max = auc_df['AUC'].max()
            auc_max_list.append(auc_max)
            best_var = auc_df[auc_df['AUC'] == auc_df['AUC'].max()]['var'].to_list()
            print('Round ' + str(n) + ', best variable: ' + str(best_var) + ' , AUC: ' + str(auc_df['AUC'].max()))
            # Add variable to selected variable list
            # if best variable categorical, pull out dummy variables for model
            if best_var[0] in input_var_categorical:
                # Generate list of dummy fields for complete fields
                best_var_formodel = generate_dummy_list(original_fieldname_list = best_var, 
                                                    full_fieldname_list = full_fieldname_list, 
                                                    reference_fieldname_list = cols_categorical_reference,
                                                    delete_reference = 'yes')
            else:
                best_var_formodel = best_var
            selected_vars = selected_vars + best_var_formodel
            # Drop best variable from list for next round
            var_list.remove(best_var[0])
            
            # Save best variable to list
            best_var_list.append(best_var)
            
        
        manual_sfs_result = pd.DataFrame({'Variable added':best_var_list,
                                          'AUC':auc_max_list})
        
        manual_sfs_result_list.append(manual_sfs_result)
        
        # -------------------------------------------------------------------------
        # Identify model with max AUC
        model_auc_max_idx = manual_sfs_result['AUC'].idxmax()
        best_model_vars = manual_sfs_result['Variable added'][0:model_auc_max_idx+1].to_list()
        best_model_vars = [item for sublist in best_model_vars for item in sublist]
        best_model_vars_list.append(best_model_vars)


#%% For models with highest AUC, generate inverse probability weights for probability of reporting infection
# -----------------------------------------------------------------------------
# Function to generate weights from model
def generate_IPW(data, model_fit, weight_colname_suffix):
    """ Generate scaled inverse probability weights from logistic regression fit and add to dataset """
    # Generate propensity scores 
    propensity_score =  model_fit.fittedvalues
    # Add to input data
    data['probability_'+weight_colname_suffix] = propensity_score
    # Plot histogram of weights
    ax = plt.figure()
    ax = sns.histplot(data=data, x='probability_'+weight_colname_suffix, hue=outcome_var, element="poly")
    ax1 = plt.figure()
    ax1 = sns.histplot(data=data, x='probability_'+weight_colname_suffix, hue=outcome_var, element="poly", stat = "probability", common_norm = False)
    
    # Following guidance in CLS report "Handling non-response in COVID-19 surveys across five national longitudinal studies" (not ATT or ATE -)
    # Calculate non-response weights as simply 1/probability of response 
    data['probability_'+weight_colname_suffix+'_inverse'] = np.divide(1, data['probability_'+weight_colname_suffix])
    # Scale weights so that sum = number of participants in sample
    scaling_factor = data.shape[0] / (data['probability_'+weight_colname_suffix+'_inverse'].sum())
    data['IPW_'+weight_colname_suffix] = scaling_factor * data['probability_'+weight_colname_suffix+'_inverse']
    
    return data


#%% Create IPW for all with COVID infection
# -----------------------------------------------------------------------------
# SET WHICH VARIABLES TO USE IN IPW MODELS
# Choose variables in model with highest AUC
var_categorical = ['EducationLevel_cat6',
                          'Biobank_LCQ_E1_StatusPrePandemicEmployment',
                          'Biobank_LCQ_A1_PrePandemicHealth',
                          'ZOE_demogs_healthcare_professional',
                          'Region',
                          'FirstLanguage_cat3',
                          'ZOE_demogs_sex',
                          ]
var_continuous = ['PRISMA7_score']

# -----------------------------------------------------------------------------
# Round 1, partial or full completion
# Set output variable
outcome_var = 'SelectionIntoAnalysisFlag'
logreg_model_var_list_forweight = [[var_continuous, var_categorical, 'NA_predictiveonly_forweight']]

# Run model
model_results_summary_covidinfection_forweight, model_auc_summary_covidinfection_forweight, model_fit_list_covidinfection_forweight = run_logistic_regression_models(data = data_0_allparticipants, 
                                                                          data_full_col_list = data_0_allparticipants_cols, 
                                                                          logreg_model_var_list = logreg_model_var_list_forweight, 
                                                                          outcome_var = outcome_var, 
                                                                          use_weights = 'no', 
                                                                          weight_var = '', 
                                                                          filter_missing = '',
                                                                          plot_model = 'Yes',
                                                                          do_poisson = do_poisson_IPW)

model_fit_covidinfection_forweight = model_fit_list_covidinfection_forweight[0]

# Generate scaled inverse probability weights from model fit
data_0_allparticipants = generate_IPW(data = data_0_allparticipants,
                                             model_fit = model_fit_covidinfection_forweight,
                                             weight_colname_suffix = 'SelectionIntoAnalysis')

test = data_0_allparticipants[['IPW_SelectionIntoAnalysis',
                                      ]]

# Plot histogram of weights
xlims = [0, 3]
titlelabel = 'Round 1, full or partial completion'
legend_offset = -0.5

ax = plt.figure()
ax = sns.histplot(data=data_0_allparticipants, x='IPW_SelectionIntoAnalysis', hue=outcome_var, element="poly")
ax.set_xlim(xlims[0], xlims[1])
ax.set_xlabel('Inverse probability of selection into analysis weight')
ax.set_title(titlelabel)
sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, legend_offset), 
                title='Participation: 0 = Not selected, 1 = Selected')

ax1 = plt.figure()
ax1 = sns.histplot(data=data_0_allparticipants, x='IPW_SelectionIntoAnalysis', hue=outcome_var, element="poly", stat = "probability", common_norm = False)
ax1.set_xlim(xlims[0], xlims[1])
ax1.set_xlabel('Inverse probability of selection into analysis weight')
ax1.set_ylabel('Normalised count')
ax1.set_title(titlelabel)
sns.move_legend(ax1, "lower center", bbox_to_anchor=(0.5, legend_offset), 
                title='Participation: 0 = Not selected, 1 = Selected')

# -----------------------------------------------------------------------------
# Generate weights table
# -----------------------------------------------------------------------------
weights_selectionintoanalysis = data_0_allparticipants[['cssbiobank_id',
                                                'IPW_SelectionIntoAnalysis',
                                                ]].copy()

# -----------------------------------------------------------------------------
# Merge weight into analysis sample
data_1a_anyevidence_symptomatic = pd.merge(data_1a_anyevidence_symptomatic, weights_selectionintoanalysis, how = 'left',  on = 'cssbiobank_id')

# -----------------------------------------------------------------------------
# Multiply participation weight with selection into analysis weight to generate 'master' weight for analysis
data_1a_anyevidence_symptomatic['IPW_ParticipationPlusSelection'] = data_1a_anyevidence_symptomatic['IPW_Participation_LCQ2022_Any_stringencylimit14'] * data_1a_anyevidence_symptomatic['IPW_SelectionIntoAnalysis']

# Scale weights so that sum = number of participants in sample
scaling_factor = data_1a_anyevidence_symptomatic.shape[0] / (data_1a_anyevidence_symptomatic['IPW_ParticipationPlusSelection'].sum())
data_1a_anyevidence_symptomatic['IPW_ParticipationPlusSelection'] = scaling_factor * data_1a_anyevidence_symptomatic['IPW_ParticipationPlusSelection']

data_1a_anyevidence_symptomatic['IPW_ParticipationPlusSelection'].sum()

test = data_1a_anyevidence_symptomatic[['cssbiobank_id','IPW_Participation_LCQ2022_Any_stringencylimit14','IPW_SelectionIntoAnalysis','IPW_ParticipationPlusSelection']]
test_describe = test.describe()

# Winsorise weights to reduce importance of outliers (e.g. limit weight at value of 5th or 9th percentile)
# Calculate limits based on 5th and 95th percentiles among those with full participation
winsor_values = data_1a_anyevidence_symptomatic['IPW_ParticipationPlusSelection'].quantile(q = [0.01, 0.05, 0.95, 0.99])

do_winsorisation = 1
if do_winsorisation == 1:
    # Select columns to do winsorization on
    weight_winsorization_list = [#
                                 'IPW_ParticipationPlusSelection',
                                 ]
    # manual limits based on 5th and 95th percentiles among those with full participation
    manual_limit_list = [[winsor_values[0.05], winsor_values[0.95]]]
    # Do winsorization
    data_1a_anyevidence_symptomatic = winsorization(data = data_1a_anyevidence_symptomatic, 
                         winsorization_limits = [0.05,0.95],
                         winsorization_col_list = weight_winsorization_list,
                         set_manual_limits = 'yes',
                         manual_limit_list = manual_limit_list)
    
    # Re-scale again after winsorising
    # Scale weights so that sum = number of participants in sample
    scaling_factor = data_1a_anyevidence_symptomatic.shape[0] / (data_1a_anyevidence_symptomatic['IPW_ParticipationPlusSelection' + '_winsorised'].sum())
    data_1a_anyevidence_symptomatic['IPW_ParticipationPlusSelection' + '_winsorised'] = scaling_factor * data_1a_anyevidence_symptomatic['IPW_ParticipationPlusSelection' + '_winsorised']
    
    test = data_1a_anyevidence_symptomatic[['cssbiobank_id','IPW_ParticipationPlusSelection', 'IPW_ParticipationPlusSelection' + '_winsorised']]
    test_describe = test.describe()


#%% Create IPW for self-report/diagnosed Long COVID
# -----------------------------------------------------------------------------
# SET WHICH VARIABLES TO USE IN IPW MODELS
# Choose variables in model with highest AUC
var_categorical = ['ZOE_demogs_healthcare_professional', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'FirstLanguage_cat3', 'Combined_Age_2022_grouped_decades', 'Combined_BMI_cat5',
                   ]
var_continuous = []

# -----------------------------------------------------------------------------
# Round 1, partial or full completion
# Set output variable
outcome_var = 'SelectionIntoAnalysis_SelfReportLongCOVID_Flag'
logreg_model_var_list_forweight = [[var_continuous, var_categorical, 'NA_predictiveonly_forweight']]

# Run model
model_results_summary_covidinfection_forweight, model_auc_summary_covidinfection_forweight, model_fit_list_covidinfection_forweight = run_logistic_regression_models(data = data_0_allparticipants, 
                                                                          data_full_col_list = data_0_allparticipants_cols, 
                                                                          logreg_model_var_list = logreg_model_var_list_forweight, 
                                                                          outcome_var = outcome_var, 
                                                                          use_weights = 'no', 
                                                                          weight_var = '', 
                                                                          filter_missing = '',
                                                                          plot_model = 'Yes',
                                                                          do_poisson = do_poisson_IPW)

model_fit_covidinfection_forweight = model_fit_list_covidinfection_forweight[0]

# Generate scaled inverse probability weights from model fit
data_0_allparticipants = generate_IPW(data = data_0_allparticipants,
                                             model_fit = model_fit_covidinfection_forweight,
                                             weight_colname_suffix = 'SelectionIntoAnalysis_SelfReportLongCOVID')

test = data_0_allparticipants[['IPW_SelectionIntoAnalysis_SelfReportLongCOVID',
                                      ]]

# Plot histogram of weights
xlims = [0, 3]
titlelabel = 'Round 1, full or partial completion'
legend_offset = -0.5

ax = plt.figure()
ax = sns.histplot(data=data_0_allparticipants, x='IPW_SelectionIntoAnalysis_SelfReportLongCOVID', hue=outcome_var, element="poly")
ax.set_xlim(xlims[0], xlims[1])
ax.set_xlabel('Inverse probability of selection into analysis weight')
ax.set_title(titlelabel)
sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, legend_offset), 
                title='Participation: 0 = Not selected, 1 = Selected')

ax1 = plt.figure()
ax1 = sns.histplot(data=data_0_allparticipants, x='IPW_SelectionIntoAnalysis_SelfReportLongCOVID', hue=outcome_var, element="poly", stat = "probability", common_norm = False)
ax1.set_xlim(xlims[0], xlims[1])
ax1.set_xlabel('Inverse probability of selection into analysis weight')
ax1.set_ylabel('Normalised count')
ax1.set_title(titlelabel)
sns.move_legend(ax1, "lower center", bbox_to_anchor=(0.5, legend_offset), 
                title='Participation: 0 = Not selected, 1 = Selected')

# -----------------------------------------------------------------------------
# Generate weights table
# -----------------------------------------------------------------------------
weights_selectionintoanalysis = data_0_allparticipants[['cssbiobank_id',
                                                'IPW_SelectionIntoAnalysis_SelfReportLongCOVID',
                                                ]].copy()

# -----------------------------------------------------------------------------
# Merge weight into analysis sample
data_2c_anyevidence_selfreportlongcovid = pd.merge(data_2c_anyevidence_selfreportlongcovid, weights_selectionintoanalysis, how = 'left',  on = 'cssbiobank_id')

# -----------------------------------------------------------------------------
# Multiply participation weight with selection into analysis weight to generate 'master' weight for analysis
data_2c_anyevidence_selfreportlongcovid['IPW_ParticipationPlusSelection'] = data_2c_anyevidence_selfreportlongcovid['IPW_Participation_LCQ2022_Any_stringencylimit14'] * data_2c_anyevidence_selfreportlongcovid['IPW_SelectionIntoAnalysis_SelfReportLongCOVID']

# Scale weights so that sum = number of participants in sample
scaling_factor = data_2c_anyevidence_selfreportlongcovid.shape[0] / (data_2c_anyevidence_selfreportlongcovid['IPW_ParticipationPlusSelection'].sum())
data_2c_anyevidence_selfreportlongcovid['IPW_ParticipationPlusSelection'] = scaling_factor * data_2c_anyevidence_selfreportlongcovid['IPW_ParticipationPlusSelection']

data_2c_anyevidence_selfreportlongcovid['IPW_ParticipationPlusSelection'].sum()

test = data_2c_anyevidence_selfreportlongcovid[['cssbiobank_id','IPW_Participation_LCQ2022_Any_stringencylimit14','IPW_SelectionIntoAnalysis_SelfReportLongCOVID','IPW_ParticipationPlusSelection']]
test_describe = test.describe()

# Winsorise weights to reduce importance of outliers (e.g. limit weight at value of 5th or 9th percentile)
# Calculate limits based on 5th and 95th percentiles among those with full participation
winsor_values = data_2c_anyevidence_selfreportlongcovid['IPW_ParticipationPlusSelection'].quantile(q = [0.01, 0.05, 0.95, 0.99])

do_winsorisation = 1
if do_winsorisation == 1:
    # Select columns to do winsorization on
    weight_winsorization_list = [#
                                 'IPW_ParticipationPlusSelection',
                                 ]
    # manual limits based on 5th and 95th percentiles among those with full participation
    manual_limit_list = [[winsor_values[0.05], winsor_values[0.95]]]
    # Do winsorization
    data_2c_anyevidence_selfreportlongcovid = winsorization(data = data_2c_anyevidence_selfreportlongcovid, 
                         winsorization_limits = [0.05,0.95],
                         winsorization_col_list = weight_winsorization_list,
                         set_manual_limits = 'yes',
                         manual_limit_list = manual_limit_list)
    
    # Re-scale again after winsorising
    # Scale weights so that sum = number of participants in sample
    scaling_factor = data_2c_anyevidence_selfreportlongcovid.shape[0] / (data_2c_anyevidence_selfreportlongcovid['IPW_ParticipationPlusSelection' + '_winsorised'].sum())
    data_2c_anyevidence_selfreportlongcovid['IPW_ParticipationPlusSelection' + '_winsorised'] = scaling_factor * data_2c_anyevidence_selfreportlongcovid['IPW_ParticipationPlusSelection' + '_winsorised']
    
    test = data_2c_anyevidence_selfreportlongcovid[['cssbiobank_id','IPW_ParticipationPlusSelection', 'IPW_ParticipationPlusSelection' + '_winsorised']]
    test_describe = test.describe()
    
    


#%% Export dataset for MAIHDA intersectional models in R
select_cols = ['Combined_Age_2022_grouped_decades', 'Combined_Ethnicity_cat2', 'Combined_EthnicityCategory', # control variables
               'EducationLevel_cat2','EducationLevel_cat3', 'EducationLevel_forstrata_cat3',
               'Region', 'RUC_Latest2023',
               'Combined_IMD_Quintile','Combined_IMD_cat3',
               'ZOE_demogs_sex', # individual variables in strata
               'strata_sex_edu2_IMD5', # strata variables
               'IPW_ParticipationPlusSelection' + '_winsorised', # IPW variable
               'LongCovidDiagnosis', # Long covid
               'CovidRecovery', # Outcome variable
               'PRISMA7_cat2', 'Combined_BMI_cat5','ZOE_conditions_condition_count_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', # pre-pandemic health
               ]
data_export_1a = data_1a_anyevidence_symptomatic[select_cols].copy()
data_export_2c = data_2c_anyevidence_selfreportlongcovid[select_cols].copy()
if export_csv == 1:
    data_1a_anyevidence_symptomatic[select_cols].to_csv(r"Biobank_LCQ2022_anyevidence_symptomatic_FilterForMAIHDA.csv")
    data_2c_anyevidence_selfreportlongcovid[select_cols].to_csv(r"Biobank_LCQ2022_anyevidence_selfreportlongcovid_FilterForMAIHDA.csv")
    
    


#%% Run logistic regression models for inference purposes, including adjustment sets as suggested by proposed DAG
# -----------------------------------------------------------------------------
#%% Create variable sets for models
# -----------------------------------------------------------------------------
# List of lists containing permutations of variables to create models for
# list structure: [1. continuous variables, 2. categorical variables, 3. Name of exposure variable]

# Treatment and Recovery have same adjustment sets for all exposures
# Treatment, Recovery and Diagnosis have same adjustment sets for all exposures up to Infection wave, then diverges for "Column 3: During pandemic". 

# ----------------------------------------------------------------------------
# Split into common groups of variable sets
### Adjustment for column 1, 2, + infection wave, same for all subsamples
logreg_model_var_list_column_1_2_infectionwave = [
                         
                         # --------------------------------------------------
                         # COLUMN 1: Data set in early life
                         # Age - Sex only
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex',
                              ],'Combined_Age_2022_grouped_decades'],
                           # Sex - Age only
                           [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex',
                              ],'ZOE_demogs_sex'],
                           # Ethnicity - Age only
                           [[],['Combined_Age_2022_grouped_decades', 'Combined_EthnicityCategory',
                              ],'Combined_EthnicityCategory'],
                           # First language - Age, Sex, Ethnic group
                           [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3'],'FirstLanguage_cat3'],
                         
                           ### Age, Sex, Ethnicity, First language
                           # Education level - full
                           [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'EducationLevel_cat6',
                              'FirstLanguage_cat3',
                              ],'EducationLevel_cat6'],
                           
                           # --------------------------------------------------
                           # COLUMN 2: Data collected at start of pandemic (some exceptions but assumed to not change)                           
                           ### Age, Sex, Ethnicity, First language, Education
                           # Region
                           [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region',], 'Region'],
                           ### Age, Sex, Ethnicity, First language, Education, Region
                           # Rural-urban classification
                           [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'RUC_Latest2023',], 'RUC_Latest2023'],
                           
                           
                           ### Age, Sex, Ethnicity, First language, Education, Region
                           # Pre-pandemic employment status
                           [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'RUC_Latest2023', 'Biobank_LCQ_E1_StatusPrePandemicEmployment',], 'Biobank_LCQ_E1_StatusPrePandemicEmployment'],
                           
                           ### Age, Sex, Ethnicity, First language, Education, Region, Pre-pandemic employment status                  
                           # Deprivation - decile
                           [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'RUC_Latest2023', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_Decile',], 'Combined_IMD_Decile'],                           
                          
                          ### Age, Sex, Ethnicity, First language, Education, Region, Pre-pandemic employment status, Deprivation
                          # Physical health conditions count
                          [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'RUC_Latest2023', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'ZOE_conditions_condition_count_cat3',], 'ZOE_conditions_condition_count_cat3'],
                          # Body mass index
                          [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'RUC_Latest2023', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Combined_BMI_cat5',], 'Combined_BMI_cat5'],
                          # PRISMA-7 frailty assessment
                          [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'RUC_Latest2023', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'PRISMA7_cat2',], 'PRISMA7_cat2'],
                          # Pre-pandemic general health
                          [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'RUC_Latest2023', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth',], 'Biobank_LCQ_A1_PrePandemicHealth'],                          
                         
                         ]



# ----------------------------------------------------------------------------
# EXPOSURES: Column 3 + Long covid diagnosis
# FOR SAMPLES: LCQ 2022 subsamples (SO CERTAIN VARIABLES WITH MISSING DATA E.G. 2021 LCQ DATA NOT INCLUDED HERE)
# FOR OUTCOMES: RECOVERY 

logreg_model_var_list_column_3_longcoviddiagnosis_LCQ2022 = [
                         ### DON'T INCLUDE COVID-19 ILLNESS CHARACTERISTICS - SAVE THIS FOR MEDIATION MODELS
                         ### DON'T INCLUDE OTHER CONTEMPORANEOUS SOCIO-DEMOGS VARIABLES TO SEE TOTAL EFFECT OF EXPOSURE 
                         
                         # ----------------------------------------------------
                         # COVID-19 ILLNESS FACTORS - ACUTE (BEFORE PANDEMIC SOCIAL FACTORS)
                         ### Age, Sex, Ethnicity, First language, Education, Region, Pre-pandemic employment status, Deprivation, Pre-pandemic health (general health, BMI, physical condition count, PRISMA-7)
                         # Wave/period at time of infection
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period',], 'SingleLongest_AnyEvidence_Infection_Period'], 
                         ### Age, Sex, Ethnicity, First language, Education, Region, Pre-pandemic employment status, Deprivation, Pre-pandemic health (general health, BMI, physical condition count, PRISMA-7), Wave/period at time of infection
                         # Sought urgent care during covid illness
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'SingleLongest_AnyEvidence_Infection_MedicalHelp_UrgentCare'], 'SingleLongest_AnyEvidence_Infection_MedicalHelp_UrgentCare'],
                         # Unable to access care during covid illness
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'SingleLongest_AnyEvidence_Infection_MedicalHelp_Unable'], 'SingleLongest_AnyEvidence_Infection_MedicalHelp_Unable'],
                         
                         
                         # -----------------------------------------------------
                         ## Mental health questionnaire collected during pandemic (Feb-May 2021)
                         ### Age, Sex, Ethnicity, First language, Education, Region, Pre-pandemic employment status, Deprivation, Pre-pandemic health (general health, BMI, physical condition count, PRISMA-7), Wave/period at time of infection,  Sought urgent care during covid illness
                         # Number of mental health diagnoses - moved from pre-pandemic as collected during pandemic
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4'], 'ZOE_mentalhealth_condition_cat4'],
                         
                         # ----------------------------------------------------
                         # PANDEMIC SOCIAL FACTORS
                         ### Age, Sex, Ethnicity, First language, Education, Region, Pre-pandemic employment status, Deprivation, Pre-pandemic health (general health, BMI, physical condition count, PRISMA-7), Wave/period at time of infection,  Sought urgent care during covid illness, Mental health diagnoses
                         # ADVERSE EXPERIENCES - INDIVIDUAL
                         # Lost job
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_A_LostJob'], 'CovidExperiences_A_LostJob'],
                         # Furlough
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_B_Furlough'], 'CovidExperiences_B_Furlough'],
                         # Unable to pay bills
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_C_UnableToPayBills'], 'CovidExperiences_C_UnableToPayBills'],
                         # Lost accomodation
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_D_LostAccomodation'], 'CovidExperiences_D_LostAccomodation'],
                         # Unable to afford food
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_E_UnableToAffordFood'], 'CovidExperiences_E_UnableToAffordFood'],
                         # Unable to access medication
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_F_UnableAccessMedication'], 'CovidExperiences_F_UnableAccessMedication'],
                         # Unable to access community health care
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_G_UnableAccessCommunityCare'], 'CovidExperiences_G_UnableAccessCommunityCare'],
                         # Unable to access social care
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_H_UnableAccessSocialCare'], 'CovidExperiences_H_UnableAccessSocialCare'],
                         # Unable to access hospital appointment
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_I_UnableAccessHospitalAppt'], 'CovidExperiences_I_UnableAccessHospitalAppt'],
                         # Unable to access mental health appointment
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_J_UnableAccessMHAppt'], 'CovidExperiences_J_UnableAccessMHAppt'],
                         # COVID-19 bereavement
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_K_CovidBereavement'], 'CovidExperiences_K_CovidBereavement'],
                                                  
                         # ADVERSE EXPERIENCES - GROUPED
                         # Economic difficulty during pandemic
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_Economic_Count_cat2'], 'CovidExperiences_Economic_Count_cat2'],
                         # Unable to access health and social care during pandemic
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_HealthSocialCare_Count_cat4'], 'CovidExperiences_HealthSocialCare_Count_cat4'],
                         # Any difficulty during pandemic
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_All_Count_cat5'], 'CovidExperiences_All_Count_cat5'],
                         
                         
                         # ----------------------------------------------------
                         ### Age, Sex, Ethnicity, First language, Education, Region, Pre-pandemic employment status, Deprivation, Pre-pandemic health (general health, BMI, physical condition count, PRISMA-7), Wave/period at time of infection,  Sought urgent care during covid illness, Mental health diagnoses, Pandemic experiences (use overall count)
                         # Current employment status
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_All_Count_cat5', 'EmploymentStatus'], 'EmploymentStatus'],
                         
                         ### Age, Sex, Ethnicity, First language, Education, Region, Pre-pandemic employment status, Deprivation, Pre-pandemic health (general health, BMI, physical condition count, PRISMA-7), Wave/period at time of infection,  Sought urgent care during covid illness, Mental health diagnoses, Pandemic experiences (use overall count), Current employment status
                         # Current household income
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_All_Count_cat5', 'EmploymentStatus', 'HouseholdIncome_cat9'], 'HouseholdIncome_cat9'],
                         
                         
                         # ----------------------------------------------------
                         # COVID-19 ILLNESS FACTORS - LONG-TERM (AFTER PANDEMIC SOCIAL FACTORS)
                         ### Age, Sex, Ethnicity, First language, Education, Region, Pre-pandemic employment status, Deprivation, Pre-pandemic health (general health, BMI, physical condition count, PRISMA-7), Wave/period at time of infection,  Sought urgent care during covid illness, Mental health diagnoses, Pandemic experiences (use overall count), Current employment status, Current household income
                         # Affected functioning duration
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_All_Count_cat5', 'EmploymentStatus', 'HouseholdIncome_cat9', 'FunctionalImpairmentDuration_grouped'], 'FunctionalImpairmentDuration_grouped'],
                         # Symptom duration - retrospective from self-report
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_All_Count_cat5', 'EmploymentStatus', 'HouseholdIncome_cat9', 'SingleLongest_AnyEvidence_Infection_SymptomDuration'], 'SingleLongest_AnyEvidence_Infection_SymptomDuration'],
                         # Symptom duration  - prospective from CSS
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_All_Count_cat5', 'EmploymentStatus', 'HouseholdIncome_cat9', 'symptomduration_only_grouped_stringencylimit'+stringency], 'symptomduration_only_grouped_stringencylimit'+stringency],
                         # Number of new conditions (as result of covid)
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_All_Count_cat5', 'EmploymentStatus', 'HouseholdIncome_cat9', 'NewCondition_DueToCovid_Count_cat4'], 'NewCondition_DueToCovid_Count_cat4'],
                         
                         ### Age, Sex, Ethnicity, First language, Education, Region, Pre-pandemic employment status, Deprivation, Pre-pandemic health (general health, BMI, physical condition count, PRISMA-7), Wave/period at time of infection,  Sought urgent care during covid illness, Mental health diagnoses, Pandemic experiences (use overall count), Current employment status, Current household income, Symptom duration - retrospective from self-report
                         # Diagnosed with long covid
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_All_Count_cat5', 'EmploymentStatus', 'HouseholdIncome_cat9', 'SingleLongest_AnyEvidence_Infection_SymptomDuration', 'LongCovidDiagnosis'], 'LongCovidDiagnosis'],
                         

                         ]

# ----------------------------------------------------------------------------
# EXPOSURES: Long covid treatment 
# FOR SAMPLES: LCQ 2022 subsamples (SO CERTAIN VARIABLES WITH MISSING DATA E.G. 2021 LCQ DATA NOT INCLUDED HERE)
# FOR OUTCOMES: RECOVERY 
logreg_model_var_list_longcovidtreatment_LCQ2022 = [
                         ### Age, Sex, Ethnicity, First language, Education, Region, Pre-pandemic employment status, Deprivation, Pre-pandemic health (general health, BMI, physical condition count, PRISMA-7), Wave/period at time of infection,  Sought urgent care during covid illness, Mental health diagnoses, Pandemic experiences (use overall count), Current employment status, Current household income, Symptom duration - retrospective from self-report, Diagnosed with long covid
                         # Received healthcare for long symptoms
                         [[],['Combined_Age_2022_grouped_decades', 'ZOE_demogs_sex', 'Combined_Ethnicity_cat2', 'FirstLanguage_cat3', 'EducationLevel_cat3', 'Region', 'Biobank_LCQ_E1_StatusPrePandemicEmployment', 'Combined_IMD_cat3', 'Biobank_LCQ_A1_PrePandemicHealth', 'Combined_BMI_cat5', 'ZOE_conditions_condition_count_cat3', 'PRISMA7_cat2', 'SingleLongest_AnyEvidence_Infection_Period', 'ZOE_mentalhealth_condition_cat4', 'CovidExperiences_All_Count_cat5', 'EmploymentStatus', 'HouseholdIncome_cat9', 'SingleLongest_AnyEvidence_Infection_SymptomDuration', 'LongCovidDiagnosis', 'LongCovidReferral_Appointment_Count_cat2'], 'LongCovidReferral_Appointment_Count_cat2'],
                         ]


# ----------------------------------------------------------------------------
### Add blocks together for use with 2022 LCQ Samples
# Recovery, 2022 samples
logreg_model_var_list_covidrecovery_LCQ2022 = logreg_model_var_list_column_1_2_infectionwave + logreg_model_var_list_column_3_longcoviddiagnosis_LCQ2022 + logreg_model_var_list_longcovidtreatment_LCQ2022


def add_covariates(mediator_list, model_list):
    model_list_withmediators = copy.deepcopy(model_list)
    for n in range(0,len(model_list_withmediators),1):
        mediator_list_n = mediator_list.copy()
        # If any illness variables already in variable list, delete from list of variables to add
        for var in mediator_list:
            if var in model_list_withmediators[n][1]:
                mediator_list_n.remove(var)
        # Add illness variables to categorical variable list
        model_list_withmediators[n][1] += mediator_list_n
    return model_list_withmediators


# Add pre-pandemic health factors to act as confounders in sensitivity analysis where health factors -> social factors, in contrast to main analysis where social factors -> health factors
var_list_prepandemichealthfactors = ['PRISMA7_cat2', 'Combined_BMI_cat5','ZOE_conditions_condition_count_cat3',
                          'Biobank_LCQ_A1_PrePandemicHealth',]



# -----------------------------------------------------------------------------
# Testing social factors for alternative DAG where health precedes social, health -> social
logreg_model_var_list_covidrecovery_LCQ2022_healthbeforesocial = add_covariates(mediator_list = var_list_prepandemichealthfactors, model_list = logreg_model_var_list_column_1_2_infectionwave)




#%% Specify sequence of models to run in a loop
# -----------------------------------------------------------------------------
model_inference_run_list = [#
                            # -------------------------------------------------
                            ### Outcome: Recovery
                            # Sample: any evidence, all durations 
                            # NO WEIGHTS
                            ['Covid recovery (any evidence, all durations)', # model name
                              data_1a_anyevidence_symptomatic, # dataset
                              data_1a_anyevidence_symptomatic_cols, # full dataset fieldname list
                              'CovidRecovery_Yes, I am back to normal', # outcome variable
                              logreg_model_var_list_covidrecovery_LCQ2022, # input variable list
                              'no', # use weights
                              '', # weight variable
                              ],
                            # WEIGHTED
                            ['Covid recovery (any evidence, all durations) WEIGHTED Winsor', # model name
                              data_1a_anyevidence_symptomatic, # dataset
                              data_1a_anyevidence_symptomatic_cols, # full dataset fieldname list
                              'CovidRecovery_Yes, I am back to normal', # outcome variable
                              logreg_model_var_list_covidrecovery_LCQ2022, # input variable list
                              'yes', # use weights
                              'IPW_ParticipationPlusSelection' + '_winsorised', # weight variable  IPW_Participation_LCQ2022_Any_stringencylimit14
                              ],
                            
                        
                            # -------------------------------------------------
                            # WEIGHTED & HEALTH ADDED AS ADJUSTMENT FOR SOCIAL FACTORS
                            ['Covid recovery (any evidence, all durations) WEIGHTED HEALTH BEFORE SOCIAL', # model name
                              data_1a_anyevidence_symptomatic, # dataset
                              data_1a_anyevidence_symptomatic_cols, # full dataset fieldname list
                              'CovidRecovery_Yes, I am back to normal', # outcome variable
                              logreg_model_var_list_covidrecovery_LCQ2022_healthbeforesocial, # input variable list
                              'yes', # use weights
                              'IPW_ParticipationPlusSelection' + '_winsorised', # weight variable  IPW_Participation_LCQ2022_Any_stringencylimit14
                              ],
                                                                                    
                   ]

# -----------------------------------------------------------------------------
# Function to add results for reference categories
def add_reference_odds_ratio(data):
    # Add OR = 1 for reference variables
    # Sociodemogs
    reference_age = {'Variable':'Combined_Age_2022_grouped_decades_4: 50-60', 'Odds ratio':1, 'var_exposure':'Combined_Age_2022_grouped_decades'}
    reference_sex = {'Variable':'ZOE_demogs_sex_Female', 'Odds ratio':1, 'var_exposure':'ZOE_demogs_sex'}
    reference_ethnicity = {'Variable':'Combined_Ethnicity_cat2_White', 'Odds ratio':1, 'var_exposure':'Combined_Ethnicity_cat2'}
    reference_ethnicity_cat = {'Variable':'Combined_EthnicityCategory_White', 'Odds ratio':1, 'var_exposure':'Combined_EthnicityCategory'}
    reference_region = {'Variable':'Region_London', 'Odds ratio':1, 'var_exposure':'Region'}
    reference_IMD = {'Variable':'Combined_IMD_Quintile_5.0', 'Odds ratio':1, 'var_exposure':'Combined_IMD_Quintile'}
    reference_IMD_cat3 = {'Variable':'Combined_IMD_cat3_3. Decile 8-10', 'Odds ratio':1, 'var_exposure':'Combined_IMD_cat3'}
    reference_IMD_decile = {'Variable':'Combined_IMD_Decile_10.0', 'Odds ratio':1, 'var_exposure':'Combined_IMD_Decile'}
    reference_RUC = {'Variable':'RUC_Latest2023_Urban', 'Odds ratio':1, 'var_exposure':'RUC_Latest2023'}
    
    
    # Append row to the dataframe
    data = data.append(reference_age, ignore_index=True)
    data = data.append(reference_sex, ignore_index=True)
    data = data.append(reference_ethnicity, ignore_index=True)
    data = data.append(reference_ethnicity_cat, ignore_index=True)
    data = data.append(reference_region, ignore_index=True)
    data = data.append(reference_IMD, ignore_index=True)
    data = data.append(reference_IMD_cat3, ignore_index=True)
    data = data.append(reference_IMD_decile, ignore_index=True)
    data = data.append(reference_RUC, ignore_index=True)
    
    
    # LCQ 2022 sociodemogs
    reference_language = {'Variable':'FirstLanguage_cat3_English', 'Odds ratio':1, 'var_exposure':'FirstLanguage_cat3'}  # First language
    reference_education = {'Variable':'EducationLevel_cat3_2. University degree', 'Odds ratio':1, 'var_exposure':'EducationLevel_cat3'}
    reference_education_full = {'Variable':'EducationLevel_cat6_4. University degree', 'Odds ratio':1, 'var_exposure':'EducationLevel_cat6'}
    reference_employmentstatus = {'Variable':'EmploymentStatus_Employed', 'Odds ratio':1, 'var_exposure':'EmploymentStatus'}
    reference_income = {'Variable':'HouseholdIncome_cat9_6. £40,000-£49,999', 'Odds ratio':1, 'var_exposure':'HouseholdIncome_cat9'}
    reference_pandemicexperience_economicdifficulty = {'Variable':'CovidExperiences_Economic_Count_cat2_0. None', 'Odds ratio':1, 'var_exposure':'CovidExperiences_Economic_Count_cat2'}
    reference_pandemicexperience_healthcare = {'Variable':'CovidExperiences_HealthSocialCare_Count_cat4_0. None', 'Odds ratio':1, 'var_exposure':'CovidExperiences_HealthSocialCare_Count_cat4'}
    reference_pandemicexperience_employment = {'Variable':'CovidExperiences_Employment_Count_0.0', 'Odds ratio':1, 'var_exposure':'CovidExperiences_Employment_Count'}
    reference_pandemicexperience_housing = {'Variable':'CovidExperiences_Housing_Count_0.0', 'Odds ratio':1, 'var_exposure':'CovidExperiences_Housing_Count'}
    reference_pandemicexperience_personal = {'Variable':'CovidExperiences_Personal_Count_0.0', 'Odds ratio':1, 'var_exposure':'CovidExperiences_Personal_Count'}
    reference_pandemicexperience_all = {'Variable':'CovidExperiences_All_Count_cat5_0. None', 'Odds ratio':1, 'var_exposure':'CovidExperiences_All_Count_cat5'}
    
    
    reference_pandemicexperience_A = {'Variable':'CovidExperiences_A_LostJob_No', 'Odds ratio':1, 'var_exposure':'CovidExperiences_A_LostJob'}
    reference_pandemicexperience_B= {'Variable':'CovidExperiences_B_Furlough_No', 'Odds ratio':1, 'var_exposure':'CovidExperiences_B_Furlough'}
    reference_pandemicexperience_C = {'Variable':'CovidExperiences_C_UnableToPayBills_No', 'Odds ratio':1, 'var_exposure':'CovidExperiences_C_UnableToPayBills'}
    reference_pandemicexperience_D = {'Variable':'CovidExperiences_D_LostAccomodation_No', 'Odds ratio':1, 'var_exposure':'CovidExperiences_D_LostAccomodation'}
    reference_pandemicexperience_E = {'Variable':'CovidExperiences_E_UnableToAffordFood_No', 'Odds ratio':1, 'var_exposure':'CovidExperiences_E_UnableToAffordFood'}
    reference_pandemicexperience_F = {'Variable':'CovidExperiences_F_UnableAccessMedication_No', 'Odds ratio':1, 'var_exposure':'CovidExperiences_F_UnableAccessMedication'}
    reference_pandemicexperience_G = {'Variable':'CovidExperiences_G_UnableAccessCommunityCare_No', 'Odds ratio':1, 'var_exposure':'CovidExperiences_G_UnableAccessCommunityCare'}
    reference_pandemicexperience_H = {'Variable':'CovidExperiences_H_UnableAccessSocialCare_No', 'Odds ratio':1, 'var_exposure':'CovidExperiences_H_UnableAccessSocialCare'}
    reference_pandemicexperience_I = {'Variable':'CovidExperiences_I_UnableAccessHospitalAppt_No', 'Odds ratio':1, 'var_exposure':'CovidExperiences_I_UnableAccessHospitalAppt'}
    reference_pandemicexperience_J = {'Variable':'CovidExperiences_J_UnableAccessMHAppt_No', 'Odds ratio':1, 'var_exposure':'CovidExperiences_J_UnableAccessMHAppt'}
    reference_pandemicexperience_K = {'Variable':'CovidExperiences_K_CovidBereavement_No', 'Odds ratio':1, 'var_exposure':'CovidExperiences_K_CovidBereavement'}
    
    
    reference_prepandemic_employmentstatus_occupation = {'Variable':'Biobank_LCQ_E1_StatusPrePandemicEmployment_Employed', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_E1_StatusPrePandemicEmployment'}

    # Append row to the dataframe
    data = data.append(reference_language, ignore_index=True)
    data = data.append(reference_education, ignore_index=True)
    data = data.append(reference_education_full, ignore_index=True)
    data = data.append(reference_employmentstatus, ignore_index=True)
    data = data.append(reference_occupation, ignore_index=True)
    # data = data.append(reference_employmentstatus_occupation, ignore_index=True)
    data = data.append(reference_income, ignore_index=True)
    data = data.append(reference_pandemicexperience_economicdifficulty, ignore_index=True)
    data = data.append(reference_pandemicexperience_healthcare, ignore_index=True)
    data = data.append(reference_pandemicexperience_employment, ignore_index=True)
    data = data.append(reference_pandemicexperience_housing, ignore_index=True)
    data = data.append(reference_pandemicexperience_personal, ignore_index=True)
    data = data.append(reference_pandemicexperience_all, ignore_index=True)
    
    data = data.append(reference_pandemicexperience_A, ignore_index=True)
    data = data.append(reference_pandemicexperience_B, ignore_index=True)
    data = data.append(reference_pandemicexperience_C, ignore_index=True)
    data = data.append(reference_pandemicexperience_D, ignore_index=True)
    data = data.append(reference_pandemicexperience_E, ignore_index=True)
    data = data.append(reference_pandemicexperience_F, ignore_index=True)
    data = data.append(reference_pandemicexperience_G, ignore_index=True)
    data = data.append(reference_pandemicexperience_H, ignore_index=True)
    data = data.append(reference_pandemicexperience_I, ignore_index=True)
    data = data.append(reference_pandemicexperience_J, ignore_index=True)
    data = data.append(reference_pandemicexperience_K, ignore_index=True)
    
    data = data.append(reference_prepandemic_employmentstatus_occupation, ignore_index=True)   
    
    # Health related
    reference_BMI = {'Variable':'Combined_BMI_cat5_2: 18.5-25', 'Odds ratio':1, 'var_exposure':'Combined_BMI_cat5'}
    reference_comorbidities = {'Variable':'ZOE_conditions_condition_count_cat3_0 conditions', 'Odds ratio':1, 'var_exposure':'ZOE_conditions_condition_count_cat3'}
    reference_MH_conditions = {'Variable':'ZOE_mentalhealth_condition_cat4_0 conditions', 'Odds ratio':1, 'var_exposure':'ZOE_mentalhealth_condition_cat4'}
    reference_frailty = {'Variable':'PRISMA7_cat2_1. 0-2, below threshold', 'Odds ratio':1, 'var_exposure':'PRISMA7_cat2'}
    reference_prepandemichealth = {'Variable':'Biobank_LCQ_A1_PrePandemicHealth_4. Very Good', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_A1_PrePandemicHealth'}
    
    # Append row to the dataframe
    data = data.append(reference_BMI, ignore_index=True)
    data = data.append(reference_comorbidities, ignore_index=True)
    data = data.append(reference_MH_conditions, ignore_index=True)
    data = data.append(reference_frailty, ignore_index=True)
    data = data.append(reference_prepandemichealth, ignore_index=True)
    
    
    # COVID illness related
    reference_symptomduration = {'Variable':'SingleLongest_AnyEvidence_Infection_SymptomDuration_1. Less than 2 weeks', 'Odds ratio':1, 'var_exposure':'SingleLongest_AnyEvidence_Infection_SymptomDuration'}
    reference_functionalimpairmentduration = {'Variable':'FunctionalImpairmentDuration_grouped_1. Less than 2 weeks', 'Odds ratio':1, 'var_exposure':'FunctionalImpairmentDuration_grouped'}
    reference_urgentcare = {'Variable':'SingleLongest_AnyEvidence_Infection_MedicalHelp_UrgentCare_0.0', 'Odds ratio':1, 'var_exposure':'SingleLongest_AnyEvidence_Infection_MedicalHelp_UrgentCare'}
    reference_numberofcaretypessought = {'Variable':'SingleLongest_AnyEvidence_Infection_MedicalHelp_Count_cat4_0. None', 'Odds ratio':1, 'var_exposure':'SingleLongest_AnyEvidence_Infection_MedicalHelp_Count_cat4'}
    reference_unabletoaccesscare = {'Variable':'SingleLongest_AnyEvidence_Infection_MedicalHelp_Unable_0.0', 'Odds ratio':1, 'var_exposure':'SingleLongest_AnyEvidence_Infection_MedicalHelp_Unable'}
    reference_infectionperiod = {'Variable':'SingleLongest_AnyEvidence_Infection_Period_1. Before 2020-12-08 (pre-vaccination, wild-type dominant)', 'Odds ratio':1, 'var_exposure':'SingleLongest_AnyEvidence_Infection_Period'}
    reference_newconditions = {'Variable':'NewCondition_DueToCovid_Count_cat4_0. None', 'Odds ratio':1, 'var_exposure':'NewCondition_DueToCovid_Count_cat4'}
    reference_CSSgroup = {'Variable':'symptomduration_grouped1_stringencylimit'+stringency+'_P6: Positive COVID-19, 12+ weeks', 'Odds ratio':1, 'var_exposure':'symptomduration_grouped1_stringencylimit'+stringency}
    reference_CSSsymptomduration = {'Variable':'symptomduration_only_grouped_stringencylimit'+stringency+'_2: 0-2 weeks', 'Odds ratio':1, 'var_exposure':'symptomduration_only_grouped_stringencylimit'+stringency}
    
    
        
    # Append row to the dataframe
    data = data.append(reference_symptomduration, ignore_index=True)
    data = data.append(reference_urgentcare, ignore_index=True)
    data = data.append(reference_functionalimpairmentduration, ignore_index=True)
    data = data.append(reference_numberofcaretypessought, ignore_index=True)
    data = data.append(reference_unabletoaccesscare, ignore_index=True)
    data = data.append(reference_infectionperiod, ignore_index=True)
    data = data.append(reference_newconditions, ignore_index=True)
    data = data.append(reference_CSSgroup, ignore_index=True)
    data = data.append(reference_CSSsymptomduration, ignore_index=True)
    
    # Long covid diagnosis & services received
    reference_longcoviddiagnosis = {'Variable':'LongCovidDiagnosis_No, but I do believe I have or have had Long COVID', 'Odds ratio':1, 'var_exposure':'LongCovidDiagnosis'}
    reference_longcovidservices = {'Variable':'LongCovidReferral_Appointment_Count_cat2_0. None', 'Odds ratio':1, 'var_exposure':'LongCovidReferral_Appointment_Count_cat2'}   
    
    data = data.append(reference_longcoviddiagnosis, ignore_index=True)
    data = data.append(reference_longcovidservices, ignore_index=True)
    
    
    # Symptoms at 12+ weeks, LCQ 2021 B14 
    reference_B14_breathing = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_Breathing_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_Breathing'}
    reference_B14_tastesmell = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_AlteredTasteSmell_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_AlteredTasteSmell'}
    reference_B14_thinking = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_Thinking_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_Thinking'}
    reference_B14_heart = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_Heart_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_Heart'}
    reference_B14_lightheaded = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_LightHeaded_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_LightHeaded'}
    reference_B14_abs = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_Abdominal_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_Abdominal'}
    reference_B14_muscle = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_MuscleInclFatigue_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_MuscleInclFatigue'}
    reference_B14_tingling = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_TinglingPain_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_TinglingPain'}
    reference_B14_mood = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_Mood_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_Mood'}
    reference_B14_sleep = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_Sleep_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_Sleep'}
    reference_B14_skinrash = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_SkinRashes_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_SkinRashes'}
    reference_B14_jointpain = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_BoneJointPain_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_BoneJointPain'}
    reference_B14_headache = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_Headaches_0.0', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_Headaches'}
    reference_B14_count = {'Variable':'Biobank_LCQ_B14_LongTermSymptoms_Count_cat_3. Three', 'Odds ratio':1, 'var_exposure':'Biobank_LCQ_B14_LongTermSymptoms_Count_cat'}
    
    data = data.append(reference_B14_breathing, ignore_index=True)
    data = data.append(reference_B14_tastesmell, ignore_index=True)
    data = data.append(reference_B14_thinking, ignore_index=True)
    data = data.append(reference_B14_heart, ignore_index=True)
    data = data.append(reference_B14_lightheaded, ignore_index=True)
    data = data.append(reference_B14_abs, ignore_index=True)
    data = data.append(reference_B14_muscle, ignore_index=True)
    data = data.append(reference_B14_tingling, ignore_index=True)
    data = data.append(reference_B14_mood, ignore_index=True)
    data = data.append(reference_B14_sleep, ignore_index=True)
    data = data.append(reference_B14_skinrash, ignore_index=True)
    data = data.append(reference_B14_jointpain, ignore_index=True)
    data = data.append(reference_B14_headache, ignore_index=True)
    data = data.append(reference_B14_count, ignore_index=True)

            
    return data



model_results_summary_list = []
model_auc_summary_list = []
model_fit_list_list = []

do_poisson_model = 'yes' # Use poisson regression instead of logistic as models coefficients more interpretable
for n in range(0, len(model_inference_run_list), 1):
    model_name = model_inference_run_list[n][0]
    dataset = model_inference_run_list[n][1]
    full_fieldname_list = model_inference_run_list[n][2]
    outcome_var = model_inference_run_list[n][3]
    logreg_model_var_list = model_inference_run_list[n][4]
    use_weights = model_inference_run_list[n][5]
    weight_var = model_inference_run_list[n][6]
    
    # -----------------------------------------------------------------------------
    # Run models
    print('Logistic regression for outcome of: ' + model_name)
    model_results_summary, model_auc_summary, model_fit_list = run_logistic_regression_models(data = dataset, 
                                                                              data_full_col_list = full_fieldname_list, 
                                                                              logreg_model_var_list = logreg_model_var_list, 
                                                                              outcome_var = outcome_var, 
                                                                              use_weights = use_weights, 
                                                                              weight_var = weight_var,
                                                                              filter_missing = 'yes',
                                                                              plot_model = '',
                                                                              do_poisson = do_poisson_model)
       
    # Add reference rows to model results
    add_reference = 1
    if add_reference == 1:
        model_results_summary = add_reference_odds_ratio(model_results_summary)
    
    # Label model
    model_results_summary['model_name'] = model_name 
    
    model_results_summary_list.append(model_results_summary)
    model_auc_summary_list.append(model_auc_summary)
    model_fit_list_list.append(model_fit_list)

model_results_summary_recovery_anyduration = model_results_summary_list[0]


#%% Join model results together and process
# Append together
model_results_combined = model_results_summary_list[0]
for n in range(1,len(model_results_summary_list),1):
    model_results_combined = model_results_combined.append(model_results_summary_list[n])
    
# Drop constant
model_results_combined = model_results_combined[~(model_results_combined['Variable'] == 'const')].reset_index(drop = True)

# Drop rows that aren't the exposure variable
model_results_combined['var_match'] = model_results_combined.apply(lambda x: x.var_exposure in x.Variable, axis = 1)
model_results_combined = model_results_combined[(model_results_combined['var_match'] == True)].reset_index(drop = True)

# Filter columns
model_results_combined_cols = model_results_combined.columns.to_list()
col_select = ['model_name', 'Variable', 'P-value', 'Odds ratio', 'OR C.I. (lower)', 'OR C.I. (upper)', 'OR C.I. error (lower)', 'OR C.I. error (upper)', 'total_count_n', 'group_count', 'outcome_count', 'Significance', 'outcome_variable']
model_results_combined_filter = model_results_combined[col_select].copy()



# -----------------------------------------------------------------------------
# Apply multiple testing adjustment to p-values of models
# Filter for all exposures testing association with same outcome variable in turn
outcome_var_list = model_results_combined_filter['model_name'].unique()
model_results_combined_filter_list = []
for var in outcome_var_list:
    model_results_combined_filter_slice = model_results_combined_filter[(model_results_combined_filter['model_name'] == var) 
                                                                        & ~(model_results_combined_filter['P-value'].isnull())].copy()
    multiple_test_correction = fdrcorrection(model_results_combined_filter_slice['P-value'], alpha=0.05, method='indep', is_sorted=False)
    model_results_combined_filter_slice['p_value_corrected'] = multiple_test_correction[1]
    model_results_combined_filter_list.append(model_results_combined_filter_slice)
model_results_pvalue_corrected = pd.concat(model_results_combined_filter_list)

model_results_combined_filter = pd.merge(model_results_combined_filter, model_results_pvalue_corrected['p_value_corrected'], how = 'left', left_index = True, right_index = True)

# Redo significance column for corrected p value
model_results_combined_filter.loc[(model_results_combined_filter['OR C.I. (lower)'] > 1.0)
                    & (model_results_combined_filter['OR C.I. (upper)'] > 1.0)
                    & (model_results_combined_filter['p_value_corrected'] < 0.05)
                    ,'Significance_p_corrected'] = 'Significant (OR > 1), *, p < 0.05'
model_results_combined_filter.loc[(model_results_combined_filter['OR C.I. (lower)'] > 1.0)
                    & (model_results_combined_filter['OR C.I. (upper)'] > 1.0)
                    & (model_results_combined_filter['p_value_corrected'] < 0.01)
                    ,'Significance_p_corrected'] = 'Significant (OR > 1), **, p < 0.01'
model_results_combined_filter.loc[(model_results_combined_filter['OR C.I. (lower)'] > 1.0)
                    & (model_results_combined_filter['OR C.I. (upper)'] > 1.0)
                    & (model_results_combined_filter['p_value_corrected'] < 0.001)
                    ,'Significance_p_corrected'] = 'Significant (OR > 1), ***, p < 0.001'

model_results_combined_filter.loc[(model_results_combined_filter['OR C.I. (lower)'] < 1.0)
                    & (model_results_combined_filter['OR C.I. (upper)'] < 1.0)
                    & (model_results_combined_filter['p_value_corrected'] < 0.05)
                    ,'Significance_p_corrected'] = 'Significant (OR < 1), *, p < 0.05'
model_results_combined_filter.loc[(model_results_combined_filter['OR C.I. (lower)'] < 1.0)
                    & (model_results_combined_filter['OR C.I. (upper)'] < 1.0)
                    & (model_results_combined_filter['p_value_corrected'] < 0.01)
                    ,'Significance_p_corrected'] = 'Significant (OR < 1), **, p < 0.01'
model_results_combined_filter.loc[(model_results_combined_filter['OR C.I. (lower)'] < 1.0)
                    & (model_results_combined_filter['OR C.I. (upper)'] < 1.0)
                    & (model_results_combined_filter['p_value_corrected'] < 0.001)
                    ,'Significance_p_corrected'] = 'Significant (OR < 1), ***, p < 0.001'


# -----------------------------------------------------------------------------
# Map 
import Biobank_Recovery_Codebook
model_results_combined_filter['Variable_tidy'] = model_results_combined_filter['Variable'].map(Biobank_Recovery_Codebook.dictionary['variable_tidy_V3'])
model_results_combined_filter['y_pos_manual'] = model_results_combined_filter['Variable'].map(Biobank_Recovery_Codebook.dictionary['y_pos_manual_with_reference_v2'])


#%% Plot results - BREAK INTO CHUNKS
# -----------------------------------------------------------------------------
# 1 - Limits, 2 - height, 3 - domain for title
plot_list = [# socio-demogs
                [[60.5,-1], 12.5, -0.2, 'Pre-pandemic socio-demographics'],
                    [[113.5,64.5], 9, -0.25, 'Socio-demographics during pandemic'],
                     [[146,115], 6.5, -0.3, 'Health characteristics'],
                      [[212,156.5], 10, -0.25, 'COVID-19 illness characteristics'],
              
                    [[603,598], 2, -1.3, 'CSSB only socio-demographics'],


               ]

for n in range(0,len(plot_list),1):
    limits = plot_list[n][0]
    domain = plot_list[n][3]
    height = plot_list[n][1]
    # offset = plot_list[n][2]
    # height = (limits[0] - limits[1])/5
    offset = -0.075 - np.exp(-0.25*height) # follows 1-exp relationship. fitted from manual best fits
      
    # ALL INDIVIDUALS
    ### 1 series - Recovery among 1) ALl symptomatic infection, Weighted
    data1 = plot_OR_w_conf_int(data1 = model_results_combined_filter[(model_results_combined_filter['model_name'] == 'Covid recovery (any evidence, all durations) WEIGHTED Winsor')
                                                                # | (model_results_combined_filter['var_label'] == 'Yes')
                                                                ],
                              x_fieldname = 'Variable_tidy',
                              y_fieldname = 'Odds ratio',
                              conf_int_fieldnames = ['OR C.I. error (lower)','OR C.I. error (upper)'],
                              plot1_label = 'CSS Biobank: Self-reported COVID-19 infection',
                              xlims = [0.3, 3.3333],  # [0.03, 30] [0.3, 3.33333] 
                              ylims = limits,
                              titlelabel = 'Exposures: ' + domain + '\n CSS Biobank', 
                              width = 5.5, 
                              height = height*1.2,
                              y_pos_manual = 'yes',
                              color_list = ['C9'],
                              fontsize = 12,
                              legend_offset = offset,
                              invert_axis = 'yes',
                              x_logscale = 'yes',
                              x_major_tick = 0.5,
                              x_minor_tick = 0.25,
                              poisson_reg = do_poisson_model,
                              bold = 0
                              )   

    
    # ALL INDIVIDUALS - SENSITIVITY - HEALTH BEFORE SOCIAL
    ### 1 series - Recovery among 1) ALl symptomatic infection, Weighted
    data1, data2 = plot_OR_w_conf_int_2plots(data1 = model_results_combined_filter[model_results_combined_filter['model_name'] == 'Covid recovery (any evidence, all durations) WEIGHTED Winsor'],
                               data2 = model_results_combined_filter[model_results_combined_filter['model_name'] == 'Covid recovery (any evidence, all durations) WEIGHTED HEALTH BEFORE SOCIAL'],
                              x_fieldname = 'Variable_tidy',
                              y_fieldname = 'Odds ratio',
                              conf_int_fieldnames = ['OR C.I. error (lower)','OR C.I. error (upper)'],
                              plot1_label = 'Main analysis, DAG: Social factors -> Health factors',
                              plot2_label = 'Sensitivity analysis, DAG: Health factors -> Social factors',
                              xlims = [0.3, 3.3333],  # [0.03, 30]
                              ylims = limits,
                              titlelabel = 'Exposures: ' + domain + '\n CSS Biobank', 
                              width = 5.5, 
                              height = height*1.2,
                              offset = 0.2,
                              y_pos_manual = 'yes',
                              color_list = ['C9', 'C1'],
                              fontsize = 12,
                              legend_offset = offset,
                              invert_axis = 'yes',
                              x_logscale = 'yes',
                              x_major_tick = 0.5,
                              x_minor_tick = 0.25,
                              poisson_reg = do_poisson_model
                              )
     
    ### 1 series crude recovery bar chart - Recovery among 1) ALl symtpomatic infection, 2) self-reported long covid. Weighted
    data1_prop = plot_bar_1series(dataset_1 = data_1a_anyevidence_symptomatic,
                                  model_results_combined_filter = model_results_combined_filter,
                  x_fieldname = 'Variable_tidy',
                  y_fieldname = 'outcome_prop',
                  xlims = [0,1],
                    ylims = limits,
                  width = 1.2,
                  height = height*1.2,
                  y_pos_manual = 'yes',
                  color_list = ['C9', 'C2', 'C0'],
                  fontsize = 12, 
                  invert_axis = 'yes',
                  offset = 0.2, 
                  scalar = 1.8, 
                  outcome_var_categorical = 'CovidRecovery',
                  outcome_var_dummy = 'CovidRecovery_Yes, I am back to normal',
                  hide_ticks = 'yes',
                  xlabel = 'Proportion recovered'
                  )  


# ADD VARIABLE NAME LABELS
var_label_list = []

var_label_list.append({'y_pos_manual':0, 'var_label': 'Yes', 'Variable_tidy':'Age group (years)'})
var_label_list.append({'y_pos_manual':6.5, 'var_label': 'Yes', 'Variable_tidy':'Sex'})
var_label_list.append({'y_pos_manual':10, 'var_label': 'Yes', 'Variable_tidy':'Ethnic group'})
var_label_list.append({'y_pos_manual':15+(3*0.5), 'var_label': 'Yes', 'Variable_tidy':'Highest educational qualification'})
var_label_list.append({'y_pos_manual':23+(4*0.5), 'var_label': 'Yes', 'Variable_tidy':'UK Region'})
var_label_list.append({'y_pos_manual':35+(5*0.5), 'var_label': 'Yes', 'Variable_tidy':'Rural-Urban classification'})
var_label_list.append({'y_pos_manual':38+(6*0.5), 'var_label': 'Yes', 'Variable_tidy':'Pre-pandemic employment status'})
var_label_list.append({'y_pos_manual':46+(7*0.5), 'var_label': 'Yes', 'Variable_tidy':'Local area deprivation'})


var_label_list.append({'y_pos_manual':62+(7*0.5), 'var_label': 'Yes', 'Variable_tidy':'Pandemic adverse experiences'})
var_label_list.append({'y_pos_manual':77+(8*0.5), 'var_label': 'Yes', 'Variable_tidy':'Number of pandemic adverse health care experiences (of 5)'})
var_label_list.append({'y_pos_manual':83+(9*0.5), 'var_label': 'Yes', 'Variable_tidy':'Overall number of pandemic adverse experiences (of 11)'})
var_label_list.append({'y_pos_manual':90+(10*0.5), 'var_label': 'Yes', 'Variable_tidy':'Current employment status'})
var_label_list.append({'y_pos_manual':98+(11*0.5), 'var_label': 'Yes', 'Variable_tidy':'Current household income'})

var_label_list.append({'y_pos_manual':110+(12*0.5), 'var_label': 'Yes', 'Variable_tidy':'Pre-pandemic health'})
var_label_list.append({'y_pos_manual':117+(13*0.5), 'var_label': 'Yes', 'Variable_tidy':'Body mass index'})
var_label_list.append({'y_pos_manual':123+(14*0.5), 'var_label': 'Yes', 'Variable_tidy':'PRISMA-7 Frailty score'})
var_label_list.append({'y_pos_manual':127+(15*0.5), 'var_label': 'Yes', 'Variable_tidy':'Number of physical health conditions'})
var_label_list.append({'y_pos_manual':132+(16*0.5), 'var_label': 'Yes', 'Variable_tidy':'Number of mental health conditions (February 2021)'})

var_label_list.append({'y_pos_manual':149+(17*0.5), 'var_label': 'Yes', 'Variable_tidy':'Infection period'})
var_label_list.append({'y_pos_manual':154+(18*0.5), 'var_label': 'Yes', 'Variable_tidy':'Urgent care accessed during COVID-19 illness'})
var_label_list.append({'y_pos_manual':157+(19*0.5), 'var_label': 'Yes', 'Variable_tidy':'Symptom duration (retrospective self-report)'})
var_label_list.append({'y_pos_manual':166+(20*0.5), 'var_label': 'Yes', 'Variable_tidy':'Symptom duration (prospective logging)'})
var_label_list.append({'y_pos_manual':177+(21*0.5), 'var_label': 'Yes', 'Variable_tidy':'Affected function duration'})
var_label_list.append({'y_pos_manual':188+(22*0.5), 'var_label': 'Yes', 'Variable_tidy':'New conditions due to COVID-19'})
var_label_list.append({'y_pos_manual':193+(23*0.5), 'var_label': 'Yes', 'Variable_tidy':'Long COVID Diagnosis'})
var_label_list.append({'y_pos_manual':197+(24*0.5), 'var_label': 'Yes', 'Variable_tidy':'Long COVID care services received'})


# # CSSB specific
var_label_list.append({'y_pos_manual':599, 'var_label': 'Yes', 'Variable_tidy':'First language'})


for n in range(0,len(var_label_list),1):
    var_label = var_label_list[n]
    model_results_combined_filter = model_results_combined_filter.append(var_label, ignore_index=True)

# PLOT AGAIN WITH BOLD TEXT TO USE TO CUT AND PASTE VARIABLE NAMES    
for n in range(0,len(plot_list),1):
    limits = plot_list[n][0]
    domain = plot_list[n][3]
    height = plot_list[n][1]
    # offset = plot_list[n][2]
    # height = (limits[0] - limits[1])/5
    offset = -0.075 - np.exp(-0.25*height) # follows 1-exp relationship. fitted from manual best fits
    
    # ALL INDIVIDUALS - BOLD AND WITH VARIABLE LABELS FOR CUT AND PASTE JOB
    ### 1 series - Recovery among 1) ALl symptomatic infection, Weighted
    data1 = plot_OR_w_conf_int(data1 = model_results_combined_filter[(model_results_combined_filter['model_name'] == 'Covid recovery (any evidence, all durations) WEIGHTED Winsor')
                                                             | (model_results_combined_filter['var_label'] == 'Yes')
                                                                ],
                              x_fieldname = 'Variable_tidy',
                              y_fieldname = 'Odds ratio',
                              conf_int_fieldnames = ['OR C.I. error (lower)','OR C.I. error (upper)'],
                              plot1_label = 'CSS Biobank: Self-reported COVID-19 infection',
                              xlims = [0.3, 3.33333],  # [0.03, 30] [0.3, 3.33333]
                              ylims = limits,
                              titlelabel = 'Exposures: ' + domain + '\n CSS Biobank', 
                              width = 5.5, 
                              height = height*1.2,
                              y_pos_manual = 'yes',
                              color_list = ['C9'],
                              fontsize = 12,
                              legend_offset = offset,
                              invert_axis = 'yes',
                              x_logscale = 'yes',
                              x_major_tick = 0.5,
                              x_minor_tick = 0.25,
                              poisson_reg = do_poisson_model,
                              bold = 1
                              )
    

# -----------------------------------------------------------------------------
# 1 - Limits, 2 - height, 3 - domain for title
plot_list = [# socio-demogs
                [[44,-1], 8.5, -0.2, 'Pre-pandemic socio-demographics'],            


               ]
for n in range(0,len(plot_list),1):
    limits = plot_list[n][0]
    domain = plot_list[n][3]
    height = plot_list[n][1]
    # offset = plot_list[n][2]
    # height = (limits[0] - limits[1])/5
    offset = -0.075 - np.exp(-0.25*height) # follows 1-exp relationship. fitted from manual best fits
      
    # ALL INDIVIDUALS - SENSITIVITY - HEALTH BEFORE SOCIAL
    ### 1 series - Recovery among 1) ALl symptomatic infection, Weighted
    data1, data2 = plot_OR_w_conf_int_2plots(data1 = model_results_combined_filter[model_results_combined_filter['model_name'] == 'Covid recovery (any evidence, all durations) WEIGHTED'],
                               data2 = model_results_combined_filter[model_results_combined_filter['model_name'] == 'Covid recovery (any evidence, all durations) WEIGHTED HEALTH BEFORE SOCIAL'],
                              x_fieldname = 'Variable_tidy',
                              y_fieldname = 'Odds ratio',
                              conf_int_fieldnames = ['OR C.I. error (lower)','OR C.I. error (upper)'],
                              plot1_label = 'Main analysis, DAG: Social factors -> Health factors',
                              plot2_label = 'Sensitivity analysis, DAG: Health factors -> Social factors',
                              xlims = [0.28, 3.3333],  # [0.03, 30]
                              ylims = limits,
                              titlelabel = 'Exposures: ' + domain + '\n CSS Biobank', 
                              width = 5.5, 
                              height = height*1.2,
                              offset = 0.2,
                              y_pos_manual = 'yes',
                              color_list = ['C9', 'C1'],
                              fontsize = 12,
                              legend_offset = offset,
                              invert_axis = 'yes',
                              x_logscale = 'yes',
                              x_major_tick = 0.5,
                              x_minor_tick = 0.25,
                              poisson_reg = do_poisson_model
                              )
     


#%% Generate bubble plots of group sizes
def plot_groupsize_bubble(data1_prop, ylims, color, height, invert_axis, lower, upper, fontsize):
    data1_prop = data1_prop[~(data1_prop['x_manual'].isnull())]
    data1_prop['x_value'] = 1
    data1_prop['group_count_str'] = data1_prop['group_count'].astype(int).astype(str)
    data1_prop.loc[(data1_prop['group_count'] < 5), 'group_count_str'] = '< 5'
    xlims = [0.995,1.01]
    ylims = ylims
    width = 2
    height = height
    x_list = data1_prop['x_value'].to_list()
    y_list = data1_prop['x_manual'].to_list()
    value_list = data1_prop['group_count_str'].to_list() #data1_prop['group_count'].astype(int).astype(str).to_list()
    
    fig, ax = plt.subplots(figsize=(width,height))
    # plt.figure(figsize=(width,height))
    sns.scatterplot(data=data1_prop, x="x_value", y='x_manual', size="group_count", 
                    legend = False,
                    sizes=(lower, upper),
                    ax = ax, 
                    color = color)
    
    # Remove everything
    fig.patch.set_visible(False)
    ax.axis('off')
    
    for n in range(0,len(value_list),1):
        txt = value_list[n] #str(value_list[n])
        x_pos = x_list[n]
        y_pos = y_list[n]
        ax.annotate(txt, xy = (1.0012, y_pos+0.40), fontsize = fontsize)
    if len(ylims) > 0: # if y limits provided
        ax.set_ylim(ylims[0], ylims[1]) # set y axis limits
    if len(xlims) > 0: # if x limits provided
        ax.set_xlim(xlims[0], xlims[1]) # set x axis limits
        
    if invert_axis == 'yes':
        plt.gca().invert_yaxis()
        
    # Add title to use for axis label
    plt.title('Group size')
    
    return data1_prop
    
color_list = ['C9'] # ['C4','C2','C8']

# Easiest to plot all and crop, as trying to do y limits messes up relative sizes and placement of numbers
scalar = 35
data1_prop_processed = plot_groupsize_bubble(data1_prop, ylims = [], color = color_list[0], height = 80, invert_axis = 'yes', lower = data1_prop['group_count'].min()/scalar, upper = data1_prop['group_count'].max()/scalar, fontsize = 7)


#%% Generate sample characteristics tables
# Combine model results with descriptive tables produced for bubble plots
descriptive_field_list = ['Variable', 'group_count_str', 'group_prop', 'outcome_prop']
results_field_list = ['model_name', 'Variable', 'Variable_tidy', 'Odds ratio','OR C.I. (lower)','OR C.I. (upper)', 'y_pos_manual', 'P-value']
field_list = ['y_pos_manual', 'Variable', 'Variable_tidy',
              # 'Variable name', 'Category', 
              'group_count_str', 'group_count_prop_str', 'outcome_prop', 'outcome_prop_str', 'Odds ratio', 'OR C.I. (lower)', 'OR C.I. (upper)', 'P-value', 'OR_tidy_str']

# -----------------------------------------------------------------------------
### Subset 1
# Merge regression results and table produced for bubble plots
subset1_table_combined = pd.merge(data1_prop_processed[descriptive_field_list], model_results_combined_filter[model_results_combined_filter['model_name'] == 'Covid recovery (any evidence, all durations) WEIGHTED Winsor'][results_field_list], how = 'outer', on = 'Variable')

# Split Variable_tidy into variable and category
# subset1_table_combined[['Variable name','Category']] = subset1_table_combined['Variable'].str.split(': ', 1, expand=True)
# Add proportion in brackets to group count
subset1_table_combined['group_count_prop_str'] = subset1_table_combined['group_count_str'] + ' (' + ((subset1_table_combined['group_prop'] * 100).round(1)).astype(str) + '%)'

# Tidy outcome proportion as percentage
subset1_table_combined.loc[~(subset1_table_combined['outcome_prop'].isnull()), 'outcome_prop_str'] = ((subset1_table_combined['outcome_prop'] * 100).round(1)).astype(str) + '%'

# Tidy OR and confidence intervals
subset1_table_combined.loc[(subset1_table_combined['Odds ratio'] != 1), 'OR_tidy_str'] = (subset1_table_combined['Odds ratio'].round(2)).astype(str) + ' (' + (subset1_table_combined['OR C.I. (lower)'].round(2)).astype(str) + '-' + (subset1_table_combined['OR C.I. (upper)'].round(2)).astype(str) + ')'
subset1_table_combined.loc[(subset1_table_combined['Odds ratio'] == 1), 'OR_tidy_str'] = 'Reference'
# sort by y position
subset1_table_combined = subset1_table_combined.sort_values(by = 'y_pos_manual')

# filter for fields of interest
subset1_table_combined_filter = subset1_table_combined[field_list]

# Rename fields
subset1_table_combined_filter = subset1_table_combined_filter.rename(columns = {#
                                                                                # 'Variable name':'Variable_tidy',
                                    'outcome_prop':'COVID-19 recovery (proportion)',
                                    'outcome_prop_str':'COVID-19 recovery (%)',
                                    'OR C.I. (lower)':'95% CI (lower)',
                                    'OR C.I. (upper)':'95% CI (upper)',
                                    'group_count_prop_str':'N (%)',
                                    'OR_tidy_str':'Odds ratio (95% CI)',
                                    'P-value':'P-value (unadjusted)'
                                    })

y_range = [0, 700]
subset1_table_combined_filter2 = subset1_table_combined_filter[(subset1_table_combined_filter['y_pos_manual'] >= y_range[0]) 
                                                         & (subset1_table_combined_filter['y_pos_manual'] < y_range[1])]

# Apply multiple testing correction to p-values
subset1_table_combined_filter2_slice = subset1_table_combined_filter2[~(subset1_table_combined_filter2['P-value (unadjusted)'].isnull())].copy()

multiple_test_correction = fdrcorrection(subset1_table_combined_filter2_slice['P-value (unadjusted)'], alpha=0.05, method='indep', is_sorted=False) # indep for independent tests, poscorr for positively correlated tests, negcorr for negatively correlated tests
subset1_table_combined_filter2_slice['P-value (adjusted)'] = multiple_test_correction[1]

subset1_table_combined_filter2 = pd.merge(subset1_table_combined_filter2, subset1_table_combined_filter2_slice['P-value (adjusted)'], how = 'left', left_index = True, right_index = True)


for n in range(0,len(var_label_list),1):
    var_label = var_label_list[n]
    subset1_table_combined_filter2 = subset1_table_combined_filter2.append(var_label, ignore_index=True)
    

subset1_table_combined_filter2 = subset1_table_combined_filter2[(subset1_table_combined_filter2['y_pos_manual'] >= y_range[0]) 
                                                         & (subset1_table_combined_filter2['y_pos_manual'] < y_range[1])]


select_cols = ['y_pos_manual','Variable_tidy','N (%)','COVID-19 recovery (%)','Odds ratio','95% CI (lower)','95% CI (upper)','P-value (adjusted)','Odds ratio (95% CI)']
subset1_table_combined_filter3 = subset1_table_combined_filter2[select_cols]


#%% Generate sample characteristics tables
# Combine model results with descriptive tables produced for bubble plots
descriptive_field_list = ['Variable', 'group_count_str', 'group_prop', 'outcome_prop']
results_field_list = ['model_name', 'Variable', 'Variable_tidy', 'Odds ratio','OR C.I. (lower)','OR C.I. (upper)', 'y_pos_manual', 'P-value']
field_list = ['y_pos_manual', 'Variable', 'Variable name', 'Category', 'group_count_str', 'group_count_prop_str', 'outcome_prop', 'outcome_prop_str', 'Odds ratio', 'OR C.I. (lower)', 'OR C.I. (upper)', 'P-value', 'OR_tidy_str']

# -----------------------------------------------------------------------------
### Subset 1
# Merge regression results and table produced for bubble plots
subset1_table_combined = pd.merge(data1_prop_processed[descriptive_field_list], model_results_combined_filter[model_results_combined_filter['model_name'] == 'Covid recovery (any evidence, all durations) WEIGHTED'][results_field_list], how = 'outer', on = 'Variable')

# Split Variable_tidy into variable and category
subset1_table_combined[['Variable name','Category']] = subset1_table_combined['Variable_tidy'].str.split(': ', 1, expand=True)
# Add proportion in brackets to group count
subset1_table_combined['group_count_prop_str'] = subset1_table_combined['group_count_str'] + ' (' + ((subset1_table_combined['group_prop'] * 100).round(1)).astype(str) + '%)'

# Tidy outcome proportion as percentage
subset1_table_combined.loc[~(subset1_table_combined['outcome_prop'].isnull()), 'outcome_prop_str'] = ((subset1_table_combined['outcome_prop'] * 100).round(1)).astype(str) + '%'

# Tidy OR and confidence intervals
subset1_table_combined.loc[(subset1_table_combined['Odds ratio'] != 1), 'OR_tidy_str'] = (subset1_table_combined['Odds ratio'].round(2)).astype(str) + ' (' + (subset1_table_combined['OR C.I. (lower)'].round(2)).astype(str) + '-' + (subset1_table_combined['OR C.I. (upper)'].round(2)).astype(str) + ')'
subset1_table_combined.loc[(subset1_table_combined['Odds ratio'] == 1), 'OR_tidy_str'] = 'Reference'
# sort by y position
subset1_table_combined = subset1_table_combined.sort_values(by = 'y_pos_manual')

# filter for fields of interest
subset1_table_combined_filter = subset1_table_combined[field_list]

# Rename fields
subset1_table_combined_filter = subset1_table_combined_filter.rename(columns = {'Variable name':'Variable_tidy',
                                    'outcome_prop':'COVID-19 recovery (proportion)',
                                    'outcome_prop_str':'COVID-19 recovery (%)',
                                    'OR C.I. (lower)':'95% CI (lower)',
                                    'OR C.I. (upper)':'95% CI (upper)',
                                    'group_count_prop_str':'N (%)',
                                    'OR_tidy_str':'Odds ratio (95% CI)',
                                    'P-value':'P-value (unadjusted)'
                                    })

y_range = [0, 700]
subset1_table_combined_filter2 = subset1_table_combined_filter[(subset1_table_combined_filter['y_pos_manual'] >= y_range[0]) 
                                                         & (subset1_table_combined_filter['y_pos_manual'] < y_range[1])]

# Apply multiple testing correction to p-values
subset1_table_combined_filter2_slice = subset1_table_combined_filter2[~(subset1_table_combined_filter2['P-value (unadjusted)'].isnull())].copy()

multiple_test_correction = fdrcorrection(subset1_table_combined_filter2_slice['P-value (unadjusted)'], alpha=0.05, method='indep', is_sorted=False) # indep for independent tests, poscorr for positively correlated tests, negcorr for negatively correlated tests
subset1_table_combined_filter2_slice['P-value (adjusted)'] = multiple_test_correction[1]

subset1_table_combined_filter2 = pd.merge(subset1_table_combined_filter2, subset1_table_combined_filter2_slice['P-value (adjusted)'], how = 'left', left_index = True, right_index = True)


for n in range(0,len(var_label_list),1):
    var_label = var_label_list[n]
    subset1_table_combined_filter2 = subset1_table_combined_filter2.append(var_label, ignore_index=True)
    

subset1_table_combined_filter2 = subset1_table_combined_filter2[(subset1_table_combined_filter2['y_pos_manual'] >= y_range[0]) 
                                                         & (subset1_table_combined_filter2['y_pos_manual'] < y_range[1])]


select_cols = ['y_pos_manual','Variable_tidy','N (%)','COVID-19 recovery (%)','Odds ratio','95% CI (lower)','95% CI (upper)','P-value (adjusted)','Odds ratio (95% CI)']
subset1_table_combined_filter3 = subset1_table_combined_filter2[select_cols]

export_csv = 0
if export_csv == 1:
    # Export csv
    subset1_table_combined_filter2.to_csv('CovidRecovery_samplecharacteristics_CSSB.csv')

# Load TwinsUK equivalent 
subset1_table_TwinsUK = pd.read_csv(r"CovidRecovery_samplecharacteristics_TwinsUK.csv")
# Merge with TwinsUK equivalent
subset1_table_combined_merge = pd.merge(subset1_table_combined_filter2, subset1_table_TwinsUK, how = 'outer', left_on = 'y_pos_manual', right_on = 'y_pos_manual').sort_values(by = 'y_pos_manual')
    
    



