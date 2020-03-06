# Copyright 2020 Abhijit Ghosh. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
# http://www.apache.org/licenses/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelBinarizer as LBL
import numpy as np
import pandas as pd

        
def TrainVsValidSplit(X, y, test_size=0.2, random_state=123):
    """
    Create train vs test split of data
    Inputs: 
        X: list or pandas DataFrame of features
        y: list or pandas DataFrame of target
        test_size: type(float or list of size 2 (float values)): (default: 0.2)
                    if float is provided, the dataset is split into 2 parts: train and valid
                    if list is provided, the dataset will be split into 3 parts: train, valid and test
        random_state: (type: int) (default: 123) A random seed to make sure results can be regenerated

    Return:
        {
           'train_X', 
           'train_y',
           'valid_X',
           'valid_y',
           'test_X' , (returned only if test_size is a list of size 2)
           'test_y'   (returned only if test_size is a list of size 2)
        }
        
    """
    if isinstance(test_size, float):
        X_train, X_valid, y_train, y_valid = tts(X, y, test_size=test_size, random_state=random_state)
        return {
                'train_X': X_train, 
                'train_y': y_train,
                'valid_X': X_valid,
                'valid_y': y_valid
               }
        
    elif isinstance(test_size, list):
        X_train, X_valid, y_train, y_valid = tts(X, y, test_size=test_size[0], random_state=random_state)
        X_test = [None] * 2
        y_test = [None] * 2
        X_test[0], X_test[1], y_test[0], y_test[1] = tts(X_valid, y_valid, test_size=test_size[1], random_state=random_state)
        
        return {
                'train_X': X_train, 
                'train_y': y_train,
                'valid_X': X_test[0],
                'valid_y': y_test[0],
                'test_X' : X_test[1],
                'test_y' : y_test[1],
               }


def MissingValueTreatment(df, dict_of_replacements=None):
    """
    Replace Missing values in the data with the provided dictionary.
    Inputs: X (type: Pandas dataframe)
    dict_of_replacement: (type: dict) Dictionary in the format {'var1': replacement_value1, 'var2': replacement_value2, ...}
                                      replacement_value is the value you want to replace the missing value with.
                                      There are 3 unique values: mean, median and modal available for auto-replacement:
                                        'mean': will work for float and integer variables and will replace the missing values with the MEAN of the non-missing values
                                        'median': will work for float and integer variables and will replace the missing values with the MEDIAN of the non-missing values
                                        'modal': will work for string and integer variables and will replace the missing values with the value of highest occurrence.


    Returns: the dataframe with filled in missing values
    """

    X = df.copy()

    for key in dict_of_replacements.keys():
        if key in X.columns:
            if dict_of_replacements[key] in ['mean','median','modal']:
                if (dict_of_replacements[key] == 'mean') & (X[key].dtype in [float, int]):
                    dict_of_replacements[key] = X[key].mean()
                elif (dict_of_replacements[key] == 'median') & (X[key].dtype == float):
                    dict_of_replacements[key] = X[key].median()
                elif (dict_of_replacements[key] == 'modal') & (X[key].dtype in [object, int]):
                    _val_cnts_ = X[key].value_counts(dropna=True, ascending=False)
                    dict_of_replacements[key] = _val_cnts_.index[0]

            X[key].fillna(dict_of_replacements[key], inplace=True)

    return X


def XLabelBinarizer(df, columns=[]):
    """
    Create One-hot encoding of Character variables
    Inputs:
        X: DataFrame of inputs
        columns: List of columns to be converted

    Output: DataFrame with:
            - dropped raw character variables
            - Appended one-hot encoding variables with column names as 'raw_column_name_' + unique_value
    """

    X = df.copy()

    for c in columns:
        print("Processing column: " + str(c), end="\t")
        X[c] = X[c].astype(str)
        lbin = LBL()
        lbin.fit(X[c])
        _temp_ = pd.DataFrame(lbin.transform(X[c]))
        if _temp_.shape[1] > 1:
            regex_replace = " :!@#$%^&*()-=+.,?"
            classes_ = []
            for cl in lbin.classes_:
                t = cl
                for r in regex_replace:
                    t = t.replace(r, '_')
                classes_.append(t.lower())
            colnames = [c + '_' + cl for cl in classes_]
            _temp_.columns = colnames
        else:
            _temp_.columns = [c]

        _temp_.index = X.index

        X = pd.concat([X.drop(c, axis=1), _temp_], axis=1)
        print("completed")
    
    return X


def OutlierTreatment(X, method='f=1,c=99'):
    """
    Performs outlier treatment of the variables
    Inputs:
        X: Pandas DataFrame
        method: string : (default: 'f=1,c=99')
            'f=1,c=99': Floor the variable at 1st percentile and cap at 99th percentile. 1 and 99 can be replaced with the desired percentile values.
            'm=-2,m=2': Floor the variable at Mean - 2*(standard deviation) and cap at Mean + 2*(standard deviation). -2 and 2 can be replaced with the desired multipliers.
             

    Return: Treated dataframe
    """
    if isinstance(method, str):
        method = [m.split('=') for m in method.split(',')]
        if (method[0][0] == 'f') & (method[1][0]=='c'):
            print('Percentile wise floor cap:: {}'.format(method))
        elif (method[0][0] == 'm') & (method[1][0]=='m'):
            print('Mean +/- sd floor cap:: {}'.format(method))
    elif isinstance(method, dict):
        print("Dictionary passed")
    else:
        raise ValueError("'method' can be either string or dictionary")
    return X