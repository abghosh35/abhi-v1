from sklearn.model_selection import train_test_split


class data:
    """
    
    """
    
    def __init__(self):
        self.x = None
        
def TrainVsValidSplit(X, y, test_size=0.2, random_state=123):
    """
    Create train vs test split of data
    Inputs: 
        X: list or pandas DataFrame of features
        y: list or pandas DataFrame of target
    """
    if isinstance(test_size, float):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return {
                'train_X': X_train, 
                'train_y': y_train,
                'valid_X': X_valid,
                'valid_y': y_valid
               }
        
    elif isinstance(test_size, list):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size[0], random_state=random_state)
        X_test = [None] * 2
        y_test = [None] * 2
        X_test[0], X_test[1], y_test[0], y_test[1] = train_test_split(X_valid, y_valid, test_size=test_size[1], random_state=random_state)
        
        return {
                'train_X': X_train, 
                'train_y': y_train,
                'valid_X': X_test[0],
                'valid_y': y_test[0],
                'test_X' : X_test[1],
                'test_y' : y_test[1],
               }