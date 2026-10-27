import torch_geometric.transforms as T
import torch
import os
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from torch_geometric.datasets import Amazon, Planetoid, Coauthor, WikiCS, CoraFull, WebKB, Actor
from torch_geometric.transforms import ToUndirected, NormalizeFeatures, NormalizeScale, NormalizeRotation
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork, PPI
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import PygNodePropPredDataset

def load_dataset(dataset_name , dataset_dir):
    print('Dataloader: Loading Dataset', dataset_name)
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['CS','Physics']:
        dataset = Coauthor(dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['Photo','Computers']:
        dataset = Amazon(dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())   
    elif dataset_name == 'Actor':
        dataset = Actor(os.path.join(dataset_dir, "heterophilous_graph", "actor"), transform=T.NormalizeFeatures())
        dataset.name = dataset_name
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=os.path.join(dataset_dir, "heterophilous_graph"), name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['ogbn-arxiv', 'ogbn-products']:
        dataset = PygNodePropPredDataset(name=dataset_name, root=dataset_dir)
    elif dataset_name == 'dblp':
        dataset = CitationFull(dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['ppi']:
        train = PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'train')
        val = PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'val')
        test = PPI(root = dataset_dir, transform=T.NormalizeFeatures(), split = 'test')
        dataset = [train, val, test]   
    else:
        raise ValueError('Unknown dataset!!!')
    
    #print(dataset)
    print('Dataloader: Loading success.')
    print(dataset[0])
    return dataset

def get_splits(num_samples, train_ratio=0.1, valid_ratio=0.1, test_ratio=0.8, num_splits=5, seed=3456):

    assert train_ratio > 0 and valid_ratio > 0 and test_ratio > 0
    assert np.isclose(train_ratio + valid_ratio + test_ratio, 1.0), \
        "train_ratio + valid_ratio + test_ratio must be 1.0"

    splits = []
    idx = np.arange(num_samples)
    for i in range(num_splits):
        train_valid_idx, test_idx = train_test_split(idx, test_size=test_ratio, random_state=seed + i)
        valid_size_in_train_valid = valid_ratio / (train_ratio + valid_ratio)
        train_idx, valid_idx = train_test_split(train_valid_idx, test_size=valid_size_in_train_valid, random_state=seed + i)
        split = { "train": train_idx, "valid": valid_idx, "test": test_idx}
        splits.append(split)
    return splits

def eval_logistic_regression_random_splits(X, y, splits):
    one_hot_encoder = OneHotEncoder(categories='auto')
    X = normalize(X, norm='l2')
    y = y.squeeze()
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    # 如果只传入一个 split，就转成 list
    if isinstance(splits, dict):
        splits = [splits]

    accuracies = []
    f1_scores = []

    c_list = 2.0 ** np.arange(-10, 11)

    for split_id, split in enumerate(splits):
        train_idx = np.asarray(split["train"])
        valid_idx = np.asarray(split["valid"])
        test_idx = np.asarray(split["test"])

        x_train, y_train = X[train_idx], y[train_idx]
        x_valid, y_valid = X[valid_idx], y[valid_idx]
        x_test, y_test = X[test_idx], y[test_idx]

        best_valid_acc = -1.0
        best_clf = None

        for c in c_list:
            logreg = LogisticRegression(solver='liblinear', C=c, max_iter=3000)

            clf = OneVsRestClassifier(logreg, n_jobs=10)
            clf.fit(x_train, y_train)

            y_valid_pred = clf.predict_proba(x_valid)
            y_valid_pred = np.argmax(y_valid_pred, axis=1)
            y_valid_pred = one_hot_encoder.transform(
                y_valid_pred.reshape(-1, 1)
            ).astype(bool)

            valid_acc = metrics.accuracy_score(y_valid, y_valid_pred)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_clf = clf

        y_pred = best_clf.predict_proba(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        test_f1 = metrics.f1_score(y_test, y_pred, average='macro')

        accuracies.append(test_acc)
        f1_scores.append(test_f1)

    return accuracies, f1_scores

def eval_logistic_regression(X, y, data_random_seed=3456, repeat=3):
    one_hot_encoder = OneHotEncoder(categories='auto')
    # X = normalize(X, norm='l2')
    y = y.squeeze()
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    rng = np.random.RandomState(data_random_seed)
    accuracies = []
    f1_scores = []
    for _ in range(repeat):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=rng)

        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c),
                           n_jobs=10, cv=cv, verbose=0)
        clf.fit(x_train, y_train)

        y_pred = clf.predict_proba(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
        f1 = metrics.f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)
    return accuracies, f1_scores


def eval_clustering(X, y, num_classes, method="kmeans"):
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    y = y.squeeze()
    if method == "kmeans":
        km = KMeans(n_clusters=num_classes, n_init=5, random_state=0).fit(X)
        pred = km.labels_
        nmi = normalized_mutual_info_score(y, pred, average_method='arithmetic')
        ari = adjusted_rand_score(y, pred)
    else:
        dbscan = HDBSCAN().fit(X)
        pred = dbscan.labels_
        nmi = normalized_mutual_info_score(y, pred, average_method='arithmetic')
        ari = adjusted_rand_score(y, pred)

    return nmi, ari

