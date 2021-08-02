import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from tabulate import tabulate
import seaborn as sns
import warnings

sns.set(style="ticks", color_codes=True)
warnings.simplefilter('ignore')
pd.options.mode.chained_assignment = None 


class BinaryClassification:
    """
    Analyze the features and output variable.
    Inputs:
        X: pandas DataFrame: a dataframe with the full data and features
        depvar: string: (default: None) Name of the dependent variable 
        percentiles: list : (defualt: [0.01,0.05,0.5,0.95,0.99]) : Percentile points to compute
        missing_cutoff: float: (default: 0.2): A variable will tagged for dropping if it has more percentage of missing values than this cut-off 
        exclude_summary_vars: list: (default: ['std','top','freq']): Drop columns from analysis table
        max_unique_int_for_char: int : (default: 10): If an integer variable has less number of unique values than this value, then it will be considered as a categorical variable

    Usage example:
    bcr =  BinaryClassification(X, 'depvar_name')
    var_summary_df, vars_to_drop = bcr.VariableSummary() 
    /* Prints the variable summary table and also returns a Pandas DataFrame of the summary and a list of variables to be dropped based on analysis */
    
    bcr.DistributionPlot()
    /* Plots the distribution of each feature (columns) available in the dataframe X

    bcr.BivariatePlot()
    /* Plots bivariate distribution of each feature with the depvar */

    correlation = bcr.CorrelationPlot()
    /* Computes and plots the correlation between different features in the dataframe. */
    """

    def __init__(self, X, 
                 depvar=None, 
                 percentiles=[0.01,0.05,0.5,0.95,0.99],
                 missing_cutoff = 0.2,
                 exclude_summary_vars=['std','top','freq'],
                 max_unique_int_for_char=10
                ):
        if isinstance(X, np.ndarray):
            self.X = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            self.X = X
        self.depvar = depvar
        self.percentiles = percentiles
        self.missing_cutoff = missing_cutoff
        self.exclude_summary_vars = exclude_summary_vars
        self.max_unique_int_for_char = max_unique_int_for_char
        
        self.pdtabulate = lambda df:tabulate(df,headers='keys',tablefmt='psql')
        self.nrows = 5
        self.ncols = 3
        
    def VariableSummary(self):
        
        var_summary = self.X.describe(percentiles=self.percentiles, include='all').T
        var_summary['missing'] = 1 - (var_summary['count']/self.X.shape[0])
        
        var_summary['drop'] = np.where(var_summary['missing'] > self.missing_cutoff, "True (Missing cut-off reached)", None)
        var_summary['drop'] = np.where((var_summary['min'] == var_summary['max']) & pd.isnull(var_summary['drop']), "True (Min = Max)", None)
        
        
        if self.exclude_summary_vars is not None:
            cols2drop = []
            for v in self.exclude_summary_vars:
                if v in var_summary.columns:
                    cols2drop.append(v)
                    
            var_summary.drop(cols2drop, axis=1, inplace=True)

        if 'unique' in var_summary.columns:
            var_summary['unique'].fillna('', inplace=True)
        
        print("Variable summary: \n{}\n".format(self.pdtabulate(var_summary)))
        return var_summary, var_summary.loc[~pd.isnull(var_summary['drop'])].index.values
    
    
    def DistributionPlot(self):
        if self.X.shape[1] <= self.nrows * self.ncols:
            self.MakeDistributionPlot(self.X)
        else:
            max_vars = self.nrows * self.ncols
            num_splits = math.ceil(self.X.shape[1]/max_vars)
            for i in range(num_splits):
                self.MakeDistributionPlot(self.X[self.X.columns[i * max_vars: (i + 1) * max_vars]])
                
        return None
    
    def MakeDistributionPlot(self, data):
        ncols = self.ncols
        nrows = self.nrows
        fig, ax = plt.subplots(nrows, ncols, figsize=(30, 22))
        dtypes = data.dtypes
        for i in range(len(dtypes)):
            if dtypes.values[i] == object:
                data[data.columns[i]].fillna('Missing', inplace=True)
                summary = pd.DataFrame(data[data.columns[i]].value_counts(dropna=False))
                ax[i%nrows, i%ncols].bar(summary.index, summary[data.columns[i]])
            elif (dtypes.values[i] in [int, np.int64, np.int32]) & (len(np.unique(data[data.columns[i]])) < self.max_unique_int_for_char):
                data[data.columns[i]] = data[data.columns[i]].apply(lambda x: '0' + str(x) if (abs(x) < 9) else str(x))
                data[data.columns[i]].fillna('Missing', inplace=True)
                summary = pd.DataFrame(data[data.columns[i]].value_counts(dropna=False))
                ax[i%nrows, i%ncols].bar(summary.index, summary[data.columns[i]])
            else:
                ax[i%nrows, i%ncols].hist(data[data.columns[i]], bins=50)

            ax[i%nrows, i%ncols].set_title("Variable: {}".format(data.columns[i].upper()))
        plt.show()
        plt.close()
        return None
    
    def BivariatePlot(self):
        dtypes = self.X.dtypes
        data = self.X
        dtypes = pd.DataFrame(data.dtypes)
        dtypes['selected'] = dtypes[0].apply(lambda x: 1 if x in [int, np.int64, np.int32, float, np.float64, np.float32] else 0)
        data = data[dtypes.loc[dtypes['selected']==1].index]
        
        if self.depvar not in [None, '']:
            if data.shape[1] <= self.nrows * self.ncols:
                self.MakeDistributionPlot(data)
            else:
                max_vars = self.nrows * self.ncols
                num_splits = math.ceil(data.shape[1]/max_vars)
                for i in range(num_splits):
                    columns = list(data.columns[i * max_vars: (i + 1) * max_vars].values)
                    if self.depvar not in columns:
                        columns.append(self.depvar)
                    self.MakeBivariatePlot(data[columns])

        else:
            print("To make bivariate plot, 'depvar' should be specified")
                
        return None
    
    def MakeBivariatePlot(self, data):
        ncols = self.ncols
        nrows = self.nrows
        fig, ax = plt.subplots(nrows, ncols, figsize=(30, 22), sharex=True)
        
        for i in range(data.shape[1]):
            try:
                corr_coeff = np.corrcoef(data.loc[~data[data.columns[i]].isnull(), self.depvar], data.loc[~data[data.columns[i]].isnull(), data.columns[i]])[0][1]
            except:
                corr_coeff = np.nan

            sns.boxplot(x=self.depvar, y=data.columns[i], data=data, ax=ax[i%nrows, i%ncols])
            ax[i%nrows, i%ncols].set_title("Variable: {} (Correlation: {:.4f})".format(data.columns[i].upper(), corr_coeff), fontsize=15)
            ax[i%nrows, i%ncols].set_xlabel(None)
        plt.show()
        plt.close()
        
        return None
    
    
    def MakeOutlierPlot(self):    
        return None
    
    
    
    def CorrelationPlot(self):
        correlation = self.X.corr()
        plt.figure(figsize=(22, 12))
        sns.heatmap(correlation)
        plt.show()
        plt.close()
        return correlation