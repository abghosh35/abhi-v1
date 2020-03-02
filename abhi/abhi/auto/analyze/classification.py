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


class BinaryClassification:
    def __init__(self, X, 
                 depvar=None, 
                 percentiles=[0.01,0.05,0.5,0.95,0.99],
                 missing_cutoff = 0.2,
                 exclude_summary_vars=['std','top','freq']
                ):
        if isinstance(X, np.ndarray):
            self.X = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            self.X = X
        self.depvar = depvar
        self.percentiles = percentiles
        self.missing_cutoff = missing_cutoff
        self.exclude_summary_vars = exclude_summary_vars
        
        self.pdtabulate = lambda df:tabulate(df,headers='keys',tablefmt='psql')
        self.nrows = 5
        self.ncols = 3
        
    def VariableSummary(self):
        
        var_summary = self.X.describe(percentiles=self.percentiles, include='all').T
        var_summary['missing'] = 1 - (var_summary['count']/self.X.shape[0])
        
        var_summary['drop'] = np.where(var_summary['missing'] > self.missing_cutoff, "True (Missing cut-off reached)", None)
        var_summary['drop'] = np.where((var_summary['min'] == var_summary['max']) & pd.isnull(var_summary['drop']), "True (Min = Max)", None)
        
        if self.exclude_summary_vars is not None:
            var_summary.drop(self.exclude_summary_vars, axis=1, inplace=True)
        
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
        fig, ax = plt.subplots(nrows, ncols, figsize=(30, 22), sharex=True)
        dtypes = data.dtypes
        for i in range(len(dtypes)):
            if dtypes.values[i] == object:
                summary = pd.DataFrame(data[data.columns[i]].value_counts(dropna=False))
                ax[i%nrows, i%ncols].bar(summary.index, summary.values.flatten())
            elif (dtypes.values[i] in [int, np.int64, np.int32]) & (len(np.unique(data[data.columns[i]])) < 10):
                summary = pd.DataFrame(data[data.columns[i]].value_counts(dropna=False))
                ax[i%nrows, i%ncols].bar(summary.index, summary.values.flatten())
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
                
        return None
    
    def MakeBivariatePlot(self, data):
        ncols = self.ncols
        nrows = self.nrows
        fig, ax = plt.subplots(nrows, ncols, figsize=(30, 22), sharex=True)
        
        for i in range(data.shape[1]):
            try:
                corr_coeff = np.corrcoef(data[self.depvar], data[data.columns[i]])[0][1]
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
        return None