import torch as th
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def BatchGenerator(Data, batch_size=32, shuffle=True, num_workers=1):
    """
    Generate batches for training the model
    Inputs:
        Data: A tuple of X and y
        batch_size: Int (default=32) Batch size
        shuffle: Boolean (default=True) Used only for the first tuple assuming the other tuples (if available) are validation or testing data
        num_workers: Int (default=1) Number of CPU cores used to generate the batches
        
    Returns:
        A list (of size=# inputs tuple) of DataLoaders
    """
    
    Loaders = []
    for i in range(len(Data)):
        if isinstance(Data[i][1], list):
            y = np.asarray(Data[i][1])

        X = th.LongTensor(Data[i][0])
        y = th.FloatTensor(Data[i][1])
        
        Loaders.append(DataLoader(TensorDataset(X, y), 
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers
                               ))
        
        
    return Loaders