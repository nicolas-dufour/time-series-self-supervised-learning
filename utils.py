import numpy as np
import pandas as pd
import os

def load_UCR_dataset(path, dataset):
    """
    Loads the UCR dataset given in input in numpy arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    train_file = os.path.join(path, dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join(path, dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = np.expand_dims(train_array[:, 1:], 1).astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = np.expand_dims(test_array[:, 1:], 1).astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train, train_labels, test, test_labels
    # Post-publication note:
    # Using the testing set to normalize might bias the learned network,
    # but with a limited impact on the reported results on few datasets.
    # See the related discussion here: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/pull/13.
    mean = np.nanmean(np.concatenate([train, test]))
    var = np.nanvar(np.concatenate([train, test]))
    train = (train - mean) / math.sqrt(var)
    test = (test - mean) / math.sqrt(var)
    return train, train_labels, test, test_labels