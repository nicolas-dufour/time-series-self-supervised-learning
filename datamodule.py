import pandas as pd
import numpy as np
from pyts.datasets import fetch_uea_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn import preprocessing



class UnivariateTrainDataset(Dataset):
    """
    Dataset for the Univariate Time series training case. 
    This dataset sample a reference time series and a positive example
    Parameters:
    -----------
        path: str,
            Path to the data file. The time series must be in a tsv format. 
            The fist column correspond to labels and the rest to the time series
        min_length: int, Optional, default=20
            Minimum size of the subsample time series 
    """

    def __init__(self, path, min_length=20, fill_na=False):
        super().__init__()
        data = pd.read_csv(path, sep="\t", header=None)
        self.time_series = np.array(data.iloc[:, 1:])

        if fill_na:
            # print('Percentage of Nan in the training set: {:.2f}\nRemoving nan...'.format(100*np.isnan(self.time_series).sum()/(self.time_series.shape[0]*self.time_series.shape[1])))
            nan_mask = np.isnan(self.time_series)
            self.time_series[nan_mask] = np.zeros(shape=np.count_nonzero(nan_mask))
        self.min_length = min_length

    def __getitem__(self, idx):
        entire_series = self.time_series[idx]
        entire_length = entire_series.shape[0]

        pos_length = np.random.randint(self.min_length, high=entire_length + 1)

        ref_length = np.random.randint(pos_length, high=entire_length + 1)

        ref_beg = np.random.randint(0, high=entire_length + 1 - ref_length)

        pos_beg = np.random.randint(ref_beg, high=ref_beg + ref_length - pos_length + 1)

        ref_series = entire_series[ref_beg : ref_beg + ref_length]

        pos_series = entire_series[pos_beg : pos_beg + pos_length]

        return torch.FloatTensor(ref_series), torch.FloatTensor(pos_series)

    def __len__(self):
        return self.time_series.shape[0]


class UnivariateTestDataset(Dataset):
    """
    Dataset for the Univariate Time series training case. 
    This dataset sample a reference time series and a positive example
    Parameters:
    -----------
        path: str,
            Path to the data file. The time series must be in a tsv format. 
            The fist column correspond to labels and the rest to the time series
        min_length: int, Optional, default=20
            Minimum size of the subsample time series 
    """

    def __init__(self, path, min_length=20, fill_na=False):
        super().__init__()
        data = pd.read_csv(path, sep="\t", header=None)
        self.time_series = np.array(data.iloc[:, 1:])
        if fill_na:
            # print('Percentage of Nan in the test set: {:.2f}\nRemoving nan...'.format(100*np.isnan(self.time_series).sum()/(self.time_series.shape[0]*self.time_series.shape[1])))
            nan_mask = np.isnan(self.time_series)
            self.time_series[nan_mask] = np.zeros(shape=np.count_nonzero(nan_mask))
        self.labels = np.array(data.iloc[:, 0])

    def __getitem__(self, idx):
        entire_series = self.time_series[idx]
        label = self.labels[idx]

        return label, torch.FloatTensor(entire_series)

    def __len__(self):
        return self.time_series.shape[0]
    
    
class MultivariateTrainDataset(Dataset):
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
    def __init__(self, dataset, min_length=20, fill_na=False):
        super().__init__()
        
        self.time_series, _ , _, _ = fetch_uea_dataset(dataset, use_cache=True, return_X_y=True)
        
        if fill_na:
            #print('Percentage of Nan in the training set: {:.2f}\nRemoving nan...'.format(100*np.isnan(self.time_series).sum()/(self.time_series.shape[0]*self.time_series.shape[1])))
            nan_mask = np.isnan(self.time_series)
            self.time_series[nan_mask] = np.zeros(shape=np.count_nonzero(nan_mask))
        self.min_length = min_length
        
    def __getitem__(self, idx):
        entire_series = self.time_series[idx]
        entire_length = entire_series.shape[1]
        
        pos_length = np.random.randint(self.min_length, high = entire_length+1)
        
        ref_length = np.random.randint(pos_length, high = entire_length+1)
        
        ref_beg = np.random.randint(0, high = entire_length+1-ref_length)
        
        pos_beg = np.random.randint(ref_beg, high = ref_beg+ref_length-pos_length+1)
        
        ref_series = entire_series[:,ref_beg:ref_beg+ref_length]
        
        pos_series = entire_series[:,pos_beg:pos_beg+pos_length]
        
        return torch.FloatTensor(ref_series), torch.FloatTensor(pos_series)
        
    
    def __len__(self):
        return self.time_series.shape[0]
    
    
class MultivariateTestDataset(Dataset):
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
    def __init__(self, dataset, min_length=20, fill_na=False, get_train=False):
        super().__init__()  
        
        if not get_train:
            _, self.time_series , _, self.labels_bytes = fetch_uea_dataset(dataset, use_cache=False, return_X_y=True)
        else:
            self.time_series,  _ , self.labels_bytes , _ = fetch_uea_dataset(dataset, use_cache=False, return_X_y=True)
        
        if fill_na:
            nan_mask = np.isnan(self.time_series)
            self.time_series[nan_mask] = np.zeros(shape=np.count_nonzero(nan_mask))

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(self.labels_bytes)
        self.labels = label_encoder.transform(self.labels_bytes)

        
    def __getitem__(self, idx):
        entire_series = self.time_series[idx]
        label = self.labels[idx]

        return label, torch.FloatTensor(entire_series)        
    
    def __len__(self):
        return self.time_series.shape[0]


def univariate_train_collate_fn(batch):
    """
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
    """
    ref_series_list = list()
    pos_series_list = list()
    for item in batch:
        ref_series_list.append(item[0])
        pos_series_list.append(item[1])
    return (
        pad_sequence(ref_series_list, batch_first=True, padding_value=0.0),
        pad_sequence(pos_series_list, batch_first=True, padding_value=0.0),
    )




def univariate_test_collate_fn(batch):
    """
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
    """
    labels_list = list()
    series_list = list()
    for item in batch:
        labels_list.append(item[0])
        series_list.append(item[1])
    return (
        torch.LongTensor(labels_list),
        pad_sequence(series_list, batch_first=True, padding_value=0.0),
    )


def multivariate_train_collate_fn(batch):
    """
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
    """
    ref_series_list = list()
    pos_series_list = list()
    for item in batch:
        ref_series_list.append(item[0].t())
        pos_series_list.append(item[1].t())
    return (
        pad_sequence(ref_series_list, batch_first=True, padding_value=0.0).permute(0,2,1),
        pad_sequence(pos_series_list, batch_first=True, padding_value=0.0).permute(0,2,1),
    )

def multivariate_test_collate_fn(batch):
    """
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
    """
    labels_list = list()
    series_list = list()
    for item in batch:
        labels_list.append(item[0])
        series_list.append(item[1].t())
    return (
        torch.LongTensor(labels_list),
        pad_sequence(series_list, batch_first=True, padding_value=0.0).permute(0,2,1),
    )


class TimeSeriesDataModule(LightningDataModule):
    def __init__(
        self,
        train_path,
        val_path,
        batch_size,
        num_workers,
        dataset_name=None,
        min_length=20,
        fill_na=False,
        multivariate=False,
    ):
        super().__init__()
        
        #If using Multivariate dataset, we need to provide dataset_name and the root_data of where live the multivariates datasets"
        assert multivariate==False or (multivariate==True and dataset_name is not None)
        
        self.train_path = train_path
        self.test_path = val_path
        self.min_length = min_length
        self.fill_na = fill_na
        self.dataset_name = dataset_name
        self.multivariate = multivariate
        

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if self.multivariate:
            self.train_set = MultivariateTrainDataset(
                self.dataset_name, self.min_length, self.fill_na
            )
            self.val_set = MultivariateTestDataset(
                self.dataset_name, self.min_length, self.fill_na
            )
            self.test_set = MultivariateTestDataset(
                self.dataset_name, self.min_length, self.fill_na
            )
        else:
            self.train_set = UnivariateTrainDataset(
                self.train_path, self.min_length, self.fill_na
            )
            self.val_set = UnivariateTestDataset(
                self.train_path, self.min_length, self.fill_na
            )
            self.test_set = UnivariateTestDataset(
                self.test_path, self.min_length, self.fill_na
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            collate_fn=univariate_train_collate_fn if not self.multivariate else multivariate_train_collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
            collate_fn=univariate_test_collate_fn if not self.multivariate else multivariate_test_collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            shuffle=False,
            collate_fn=univariate_test_collate_fn if not self.multivariate else multivariate_test_collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    

    
