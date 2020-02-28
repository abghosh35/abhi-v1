import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

class parallel():    
    
    def __init__(self, dfname=None, function_to_apply=None, append_to_df=False, num_processes=multiprocessing.cpu_count(), return_dtype="list"):
        self.dfname = dfname
        self.function_to_apply = function_to_apply
        self.num_processes = num_processes
        self.append_to_df = append_to_df
        self.return_dtype = return_dtype
         
    def result(self):
        chunk_size = int(self.dfname.shape[0]/self.num_processes)
        chunks = [self.dfname.loc[self.dfname.index[i:(i + chunk_size)]] for i in range(0, self.dfname.shape[0], chunk_size)]
        pool = multiprocessing.Pool(processes=self.num_processes)
        __interim_result = pool.map(self.function_to_apply, chunks)

        if self.return_dtype == "list":
            result = []
            for i in __interim_result:
                result.extend(i)
        elif self.return_dtype == "DataFrame":
            result = pd.DataFrame()
            for i in __interim_result:
                result = result.append(i, ignore_index=True)
            if self.append_to_df:
                result = pd.concat([self.dfname, result], axis=1)
        else:
            result = None
            
        pool.close()

        return result