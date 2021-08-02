import numpy as np

class Regression():
    """
    Compute different metrics used for assessing a regression model
    Inputs to this function are:  
    actual: Actual value of the dependent/target variable, 
    predicted: Predicted (model output) of the target variable,
    weights (optional): Weights as will be used in computaion of weighted metrics like Weighted MAPE,
    mode (optional): This tells the function to determine which metric to return. Used values: 
                     single: Metrics will have to called manually. No metric will be returned by default., 
                     all: All the metrics will be printed. In absence of weights, actual values will be used as weights
                     list of metrics: any combination or all from the list: [MAPE, WMAPE]
                     
                     
    """
    def __init__(self, actual, predicted, weights=None, mode="single"):

        self.actual = actual
        self.predicted = predicted
        self.weights = weights if weights is not None else self.actual
        
    def MAPE(self):
        """
        MAPE: Mean Absolute Percentage Error
        Computed as: mean(|y - y_hat|/y) * 100
        
        
        """
        actual = [x if x != 0 else 1e-6 for x in self.actual]
        numerator = np.abs(np.subtract(actual, self.predicted))
        return np.mean(np.divide(numerator, actual)) * 100
    
    def WMAPE(self, weights):
        """
        WMAPE: Weighted Mean Absolute Percentage Error
        Computed as: mean((|y - y_hat| * w)/(y * w)) * 100
        where w is the weight vector.
        
        
        """
        actual = [x if x != 0 else 1e-6 for x in self.actual]
        weights = [x if x != 0 else 1e-6 for x in weights]
        abs_difference = np.abs(np.subtract(actual, self.predicted))
        numerator = np.multiply(abs_difference, weights)
        denominator = np.multiply(actual, weights)
        
        return np.mean(np.divide(numerator, denominator)) * 100