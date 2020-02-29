import numpy as np
import sklearn.metrics as mets
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

pd.set_option('display.float_format', lambda x: '%.3f' % x)

class BinaryClassificationReport:
    """
    Generates different metrics/ curves assessing the performance of the classification model
    Inputs:
        act: list Actual or True target values
        pred_probas: list Predicted probabilities of event=1
        metric: string (default="all") Which metric(s) to output
            Possible values: "all", 'accuracy', 'confusion_matrix', 'other_metrics', 'roc_auc'
            
    Additional Metric:   
        KSTable (Kolmogorov-Smirnov Statistic); Computes the traditional KS statistic table
            
    """
    def __init__(self, act, pred_probas, metric='all'):
        self.act = np.asarray(act)
        self.pred_probas = pred_probas
        self.pred = np.asarray([1 if p >= 0.5 else 0 for p in self.pred_probas])
        self.metric = metric
        self.pdtabulate = lambda df:tabulate(df,headers='keys',tablefmt='psql')
        
        if metric == "all":
            print("Accuracy: {:.2f}%\n".format(self.Accuracy() * 100))
            
            print("Confusion matrix: \n{}\n".format(self.pdtabulate(self.ConfusionMat())))
            self.OtherMetrics()
            self.RocAucScore()
        
        
    def Accuracy(self):
        return mets.accuracy_score(self.act, self.pred)
    
    def ConfusionMat(self):
        _conf_mat_orig = mets.confusion_matrix(self.act, self.pred)
        _conf_mat_ = _conf_mat_orig.astype(str)
        _conf_mat_normalized_ = np.round(mets.confusion_matrix(self.act, self.pred, normalize='true') * 100, 2)
        _conf_mat_normalized_ = np.core.defchararray.add(_conf_mat_normalized_.astype(str), '%)')
        _conf_mat_normalized_ = np.core.defchararray.add(' (', _conf_mat_normalized_)
        _conf_mat_ = np.core.defchararray.add(_conf_mat_, _conf_mat_normalized_)
        _conf_mat_ = pd.DataFrame(_conf_mat_)
        
        _conf_mat_['Total'] = _conf_mat_orig.sum(axis=1)
        _t_ = list(_conf_mat_orig.sum(axis=0))
        _t_.append(np.sum(_t_))
        _check_ = {}
        for i in range(len(_conf_mat_.columns)):
            _check_[_conf_mat_.columns[i]] = _t_[i]

        _conf_mat_ = _conf_mat_.append(_check_, ignore_index=True)
        
        indices = list(_conf_mat_.index)
        indices[-1] = ('Total')
        _conf_mat_.index= pd.MultiIndex.from_tuples([('Actual' if i != 'Total' else '', i) for i in indices])
        _conf_mat_.columns = pd.MultiIndex.from_tuples([('Predicted' if i != 'Total' else "", i) for i in _conf_mat_.columns])
        
        return _conf_mat_
    
    
    def OtherMetrics(self):
        print("Precision: {:.2f}%".format(mets.precision_score(self.act, self.pred) * 100))
        print("Recall: {:.2f}%".format(mets.recall_score(self.act, self.pred) * 100))
        print("F1-Score: {:.2f}%\n".format(mets.f1_score(self.act, self.pred) * 100))
        
        
    def RocAucScore(self):
        auc_score = mets.roc_auc_score(self.act, self.pred_probas)
        print("AUC score: {:.2f}%\n".format(auc_score * 100))
        
        fpr, tpr, _ = mets.roc_curve(self.act, self.pred_probas)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC - Area Under the Curve", fontdict={'fontsize':15})
        plt.xlabel("False Positive Rate", fontdict={'fontsize':12})
        plt.ylabel("True Positive Rate", fontdict={'fontsize':12})
        plt.legend(loc="lower right", fontsize='large')
        plt.show()
        print()
        
        
    def KSTable(self, y_true, pred_proba, bins=5, labels=None, assess_metric=None, assess_aggfunc='sum'):
        """
        Computes the traditional Kolmogorov-Smirnov Statistics to assess the seperation power
        Inputs:
            y_true: Actual value of the target variable
            pred_proba: Predicted probability of event=1
            bins: Int (default=5) Number of bins to split the probability into
            labels: list List of labels to be used for labeling the deciles. It should be in descending order or probability
            assess_metric: 
        """
        
        ks_full_mat = pd.DataFrame({'Probability': pred_proba, 'MetricVal': assess_metric, 'Event': y_true})
        ks_full_mat['ProbabilityBins'] = pd.cut(ks_full_mat['Probability'], bins=bins, right=True, include_lowest=True, precision=5)
        
        if assess_metric is not None:
            _temp_ = pd.pivot_table(ks_full_mat, index='ProbabilityBins', values=['MetricVal'], aggfunc=['count',assess_aggfunc])
        else:
            _temp_ = pd.pivot_table(ks_full_mat, index='ProbabilityBins', values=['Probability'], aggfunc=['count',assess_aggfunc])
            
        _temp_.sort_index(ascending=False, inplace=True)
        _temp_.columns = [c[0] for c in _temp_.columns]
        _temp2_ = pd.pivot_table(ks_full_mat, index='ProbabilityBins', values=['Event'], aggfunc='sum')
        _temp_ = _temp_.merge(_temp2_, how='inner', left_index=True, right_index=True)
        _temp_['NonEvent'] = _temp_['count'] - _temp_['Event']
        
        if assess_metric is not None:
            _temp_[assess_aggfunc] = _temp_[assess_aggfunc].astype(float).values.round(3)
        else:
            _temp_[assess_aggfunc] = None
            
        _temp_.rename(index=str, columns={assess_aggfunc: 'AssessMetric({})'.format(assess_aggfunc)}, inplace=True)
        _temp_['Labels'] = labels
        _temp_['EventCumSum'] = _temp_['Event'].cumsum()
        _temp_['NonEventCumSum'] = _temp_['NonEvent'].cumsum()

        total_event = _temp_['Event'].sum()
        total_nonevent = _temp_['NonEvent'].sum()
        _temp_['EventCum%'] = (_temp_['EventCumSum']/total_event * 100).round(4)
        _temp_['NonEventCum%'] = (_temp_['NonEventCumSum']/total_nonevent * 100).round(4)
        _temp_['KSMetric'] = (_temp_['EventCum%'] - _temp_['NonEventCum%']).round(2)
        
        _temp_ = _temp_[['Labels','Event','NonEvent','EventCum%','NonEventCum%','KSMetric','AssessMetric({})'.format(assess_aggfunc)]]

        print("KS (Kolmogorov-Smirnov Statistic): {:.2f}".format(_temp_['KSMetric'].max()))
        print("KS Bin#: {}\n".format(np.argmax(_temp_['KSMetric'].values) + 1))
        print("KS Table: \n{}".format(self.pdtabulate(_temp_)))
        
        return None