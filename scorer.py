from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.svm import SVC
import numpy as np


def compute_accuracy(model, datamodule):
    """
        Compute SVM accuracy score on the train then test set. Is sufficient train data, SVM C hyperparameters is found using grid search.  
        Parameters:
        -----------
            model: Pytorch Lightning Module
                The model we want to score
            datamodule: Pytorch Lightning DataModule
                The datamodule of the data we want to score
    """
    model = model.cuda()
    if not model.multivariate:
        train_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        train_emb = list()
        train_labels = list()
        for _, (labels, train_series) in enumerate(train_loader):
            train_emb.append(
                model(train_series[:, None, :].cuda()).cpu().detach().numpy()
            )
            train_labels.append(labels.numpy())
        train_emb = np.concatenate(train_emb)
        train_labels = np.concatenate(train_labels)
    else:
        train_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        train_emb = list()
        train_labels = list()
        for _, (labels, train_series) in enumerate(train_loader):
            train_emb.append(model(train_series.cuda()).cpu().detach().numpy())
            train_labels.append(labels.numpy())
        train_emb = np.concatenate(train_emb)
        train_labels = np.concatenate(train_labels)

    nb_classes = len(np.unique(train_labels))
    train_size = train_emb.shape[0]

    classifier = SVC(C=np.inf, gamma="scale")
    if train_size // nb_classes < 5 or train_size < 50:
        classifier = classifier.fit(train_emb, train_labels)
    else:
        grid_search = GridSearchCV(
            classifier,
            {
                "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf],
                "kernel": ["linear"],
                "degree": [3],
                "gamma": ["scale"],
                "coef0": [0],
                "shrinking": [True],
                "probability": [False],
                "tol": [0.001],
                "cache_size": [200],
                "class_weight": [None],
                "verbose": [False],
                "max_iter": [10000000],
                "decision_function_shape": ["ovr"],
                "random_state": [None],
            },
            cv=5,
            n_jobs=5,
        )
        if train_size <= 10000:
            grid_search.fit(train_emb, train_labels)
        else:
            # If the training set is too large, subsample 10000 train
            # examples
            split = train_test_split(
                train_emb, train_labels, train_size=10000, random_state=0, stratify=y
            )
            grid_search.fit(split[0], split[2])
        classifier = grid_search.best_estimator_

    # Cross validation score
    train_score = np.mean(
        cross_val_score(classifier, train_emb, y=train_labels, cv=5, n_jobs=5)
    )

    # Predict class for the test set
    if not model.multivariate:
        test_emb = list()
        test_labels = list()
        for _, (labels, test_series) in enumerate(test_loader):
            test_emb.append(
                model(test_series[:, None, :].cuda()).cpu().detach().numpy()
            )
            test_labels.append(labels.numpy())
        test_emb = np.concatenate(test_emb)
        test_labels = np.concatenate(test_labels)
    else:
        test_emb = list()
        test_labels = list()
        for _, (labels, test_series) in enumerate(test_loader):
            test_emb.append(model(test_series.cuda()).cpu().detach().numpy())
            test_labels.append(labels.numpy())
        test_emb = np.concatenate(test_emb)
        test_labels = np.concatenate(test_labels)

    test_score = np.mean(
        cross_val_score(classifier, test_emb, y=test_labels, cv=5, n_jobs=5)
    )

    return train_score, test_score