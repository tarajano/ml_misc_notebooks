
'''
Importing:

import sys
sys.path.insert(0, '<path to containing folder>')
import regression_functions_utils as rfu
'''


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def hello(arg='World'):
    return 'Hello {}'.format(str(arg))


def get_dtf():
    return pd.DataFrame()


def get_dummies(dtf=None, categ_cols=None, avoid_dummy_trap=True):
    ''' 
    Encodes categorical columns as dummy (binary) variables. 
    For each categorical column will remove one dummy to avoid dummy variable trap (default).
    Returns a dataframe with the encoded variables.
    '''
    _dtf = pd.DataFrame()
    
    for col in categ_cols:
        _dumm = None
        if avoid_dummy_trap:
            _dumm = pd.get_dummies(dtf[col]).iloc[:,0:-1]
        else:
            _dumm = pd.get_dummies(dtf[col])
        _dumm.columns = ['{}_{}'.format(col, c) for c in _dumm.columns]
        _dtf = pd.concat([_dtf,_dumm], axis=1)
    
    return _dtf


def compute_vif(dtf):
    """
    Implementation:
        https://etav.github.io/python/vif_factor_python.html (code below)

    Interpretation: 
        https://www.displayr.com/variance-inflation-factors-vifs/
        A value of 1 means that the predictor is not correlated with other variables.
        The higher the value, the greater the correlation of the variable with other variables.
        Values of more than 4 or 5 are sometimes regarded as being moderate to high, with values
        of 10 or more being regarded as very high. These numbers are just rules of thumb.
        Higher values signify that it is difficult to impossible to assess accurately the contribution
        of predictors to a model.

    Workarounds: 
        https://www.statisticssolutions.com/assumptions-of-multiple-linear-regression/
        If multicollinearity is found in the data, one possible solution is to center the data.
        To center the data, subtract the mean score from each observation for each independent variable.
        However, the simplest solution is to identify the variables causing multicollinearity
        issues (i.e., through correlations or VIF values) and removing those variables from the regression.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_dtf = pd.DataFrame()
    vif_dtf["features"] = dtf.columns
    vif_dtf['vif_factors'] = [variance_inflation_factor(dtf.values, i) for i in range(dtf.shape[1])]

    return vif_dtf


def shapiro_wilks_test(data):
    '''
        Normality test
        Params: <list>
        Returns a list of: statistic, p-value
    '''
    from scipy.stats import shapiro
    stat, p = shapiro(data)
    return [stat, p]


def test_H0(p_value):
    ''' Tests normality of data.
        p_value from shapiro_wilks_test()
    '''
    alpha = 0.05
    if p_value > alpha:
        print('Sample looks Gaussian (fail to reject H0 at alpha={:.3f})'.format(alpha))
    else:
        print('Sample does not look Gaussian (reject H0 at alpha={:.3f})'.format(alpha))
    print('')
    

''' Functions for generating automatically scatter plots from a dataframe '''
def _compute_multiplot_rows(lst):
    '''
        Computes the rows required to generate a multiplot figure of N rows and 2 columns. 
        lst: list of independent variable columns names in the dataframe (eg. ['xA', 'xB', 'xN'])
    '''
    COL_NUMBER = 2
    
    _lst_len = len(lst)
    _division = int(_lst_len/COL_NUMBER)
    _reminder = _lst_len%COL_NUMBER
    
    if _division == 0:
        return 1
    
    if _reminder == 0:
        return _division
    
    if _reminder > 0:
        return _division + 1
    

def get_rand_rgb_color(col_numb):
    return [np.random.rand(col_numb)]

    
def show_multi_plots(dtf, xy_dct, plt_type):
    '''
    Description
    -----------
        Generates a figure with multiple scatter plots.

    Parameters
    ----------
        dtf: A dataframe containing the independent and dependent variables.
        xy_dct: dictionary with keys x and y, specifying the column names of independent
                and dependent variables respectively.
            Example: { 'x': ['col_xA','col_xB','col_xC'], 'y': 'col_y'}
        plt_type: plot type [scatter | hist]
            
            
    TODO
    ----
        Fix a function fail/error at instruction "axs[r, c].scatter" when passed a list of two colums to plot 
        from dataframe.
    '''
    COL_NUMBER = 2

    x_idx = 0
    calpha = .7
    y_name = xy_dct['y']
    x_lst_len = len(xy_dct['x'])
    rows_number = _compute_multiplot_rows(xy_dct['x'])
    
    fig, axs = plt.subplots(rows_number, 2, figsize=(12, 12))

    for r in range (0, rows_number):
        for c in range (0, COL_NUMBER):
            if x_idx == x_lst_len: 
                break
            x_name = xy_dct['x'][x_idx]
            
            if plt_type == 'scatter':
                axs[r, c].scatter(dtf[x_name], dtf[y_name], c=get_rand_rgb_color(x_lst_len), alpha=calpha)
                axs[r, c].set_ylabel(y_name, fontsize=10)
                axs[r, c].set_xlabel(x_name, fontsize=10)
                
            if plt_type == 'hist':
                axs[r, c].hist(dtf[x_name], color=get_rand_rgb_color(x_lst_len), alpha=calpha)
            
            axs[r, c].set_title(x_name, fontsize=12)
            x_idx += 1
    
    plt.show()

    
def plot_scatter_line(x=None, y=None, y_pred=None, x_name='x', y_name='y', y_pred_name='y_pred'):
    plt.scatter(x, y, color='r', alpha=.3)
    plt.plot(x, y_pred, color='blue', alpha=.6)
    plt.title('{} vs {} and Predicted'.format(x_name, y_name))
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()

    
def plot_residuals(x=None, y=None, x_name='x', y_name='residuals'):
    plt.scatter(x, y, color='gray', alpha=.5)
    plt.hlines(0, min(x), max(x), color='black', alpha=.8)
    plt.title('Residuals')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()