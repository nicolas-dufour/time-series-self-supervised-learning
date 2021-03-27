import pandas as pd
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

class UnivariateTrainDataset(Dataset):
    '''
    Dataset for the Univariate Time series training case. 
    This dataset sample a reference time series and a positive example
    Parameters:
    -----------
        path: str,
            Path to the data file. The time series must be in a tsv format. 
            The fist column correspond to labels and the rest to the time series
        min_length: int, Optional, default=20
            Minimum size of the subsample time series 
    '''
    def __init__(self, path, min_length =20):
        super().__init__()
        data = pd.read_csv(path,sep='\t', header=None)
        self.time_series = np.array(data.iloc[:,1:])
        
        self.min_length = min_length
        
    def __getitem__(self, idx):
        entire_series = self.time_series[idx]
        entire_length = entire_series.shape[0]
        
        pos_length = np.random.randint(self.min_length, high = entire_length+1)
        
        ref_length = np.random.randint(pos_length, high = entire_length+1)
        
        ref_beg = np.random.randint(0, high = entire_length+1-ref_length)
        
        pos_beg = np.random.randint(ref_beg, high = ref_beg+ref_length-pos_length+1)
        
        ref_series = entire_series[ref_beg:ref_beg+ref_length]
        
        pos_series = entire_series[pos_beg:pos_beg+pos_length]
        
        return torch.FloatTensor(ref_series), torch.FloatTensor(pos_series)
        
    
    def __len__(self):
        return self.time_series.shape[0]

class UnivariateTestDataset(Dataset):
    '''
    Dataset for the Univariate Time series training case. 
    This dataset sample a reference time series and a positive example
    Parameters:
    -----------
        path: str,
            Path to the data file. The time series must be in a tsv format. 
            The fist column correspond to labels and the rest to the time series
        min_length: int, Optional, default=20
            Minimum size of the subsample time series 
    '''
    def __init__(self, path, min_length =20):
        super().__init__()
        data = pd.read_csv(path,sep='\t', header=None)
        self.time_series = np.array(data.iloc[:,1:])
        self.labels = np.array(data.iloc[:,0])
        self.min_length = min_length
        
    def __getitem__(self, idx):
        entire_series = self.time_series[idx]
        label = self.labels[idx]

        return label, torch.FloatTensor(entire_series)
        
    
    def __len__(self):
        return self.time_series.shape[0]
    

def univariate_train_collate_fn(batch):
    '''
    Function to create the batch for training. Used to pad the time-series so they fit in the same batch.
    Auxiliary function for the dataloader
    Parameters:
    -----------
        batch: list
            List containing the individual dataset items.
    Output:
    -------
        padded_ref_series: Torch Tensor (batch_size,max_ref_series_length)
            Padded reference series
        padded_ref_series: Torch Tensor (batch_size,max_pos_series_length)
            Padded positives series
    '''
    ref_series_list = list()
    pos_series_list = list()
    for item in batch:
        ref_series_list.append(item[0])
        pos_series_list.append(item[1])
    return pad_sequence(ref_series_list, batch_first=True, padding_value =0.0), pad_sequence(pos_series_list, batch_first=True, padding_value =0.0)


def univariate_test_collate_fn(batch):
    '''
    Function to create the batch. Used to pad the time-series so they fit in the same batch.
    Auxiliary function for the dataloader
    Parameters:
    -----------
        batch: list
            List containing the individual dataset items.
    Output:
    -------
        padded_ref_series: Torch Tensor (batch_size,max_ref_series_length)
            Padded reference series
        padded_ref_series: Torch Tensor (batch_size,max_pos_series_length)
            Padded positives series
    '''
    labels_list = list()
    series_list = list()
    for item in batch:
        labels_list.append(item[0])
        series_list.append(item[1])
    return torch.LongTensor(labels_list), pad_sequence(series_list, batch_first=True, padding_value =0.0)


class TimeSeriesDataModule(LightningDataModule):
    def __init__(self, train_path, val_path, batch_size, num_workers, min_length=20, multivariate=False):
        super().__init__()
        self.train_path = train_path
        self.test_path = val_path
        self.min_length = min_length
        self.multivariate = multivariate

        self.batch_size = batch_size
        self.num_workers = num_workers
        
    
    def setup(self):
        if self.multivariate:
            raise "Multivariate not implemented yet"
        else:
            self.train_set = UnivariateTrainDataset(
                self.train_path,
                self.min_length)
            self.val_set = UnivariateTestDataset(
                self.train_path,
                self.min_length)
            self.test_set = UnivariateTestDataset(
                self.test_path,
                self.min_length)
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            collate_fn = univariate_train_collate_fn,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
            collate_fn = univariate_test_collate_fn,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            shuffle=False,
            collate_fn = univariate_test_collate_fn,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        )