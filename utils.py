from read_file import read_all_subjects
import torch
from helpers import get_train_test_fold
import bct
import numpy as np
from read_file import get_baselines


def load_data():
    data = read_all_subjects()
    X = []
    A = []
    DX = []
    BDX = []
    for d in data:
        x = []
        adj = []
        dx = []
        for i in range(len(d['node_feature'])):
            x.append(d['node_feature'][i][:, 2])
            adj.append(d['adjacency_matrix'][i])
            dx.append(int(d['dx_label'][i]) - 1)
            if i == 0:
                BDX.append(int(d['dx_label'][i]) - 1)

        if len(x) > 0:
            X.append(torch.FloatTensor(x))
            A.append(torch.FloatTensor(adj))
            DX.append(torch.tensor(dx))

    train_fold, test_fold = get_train_test_fold(X, BDX)

    return A, X, DX, train_fold, test_fold


def load_data_autoencoder(no_feature=False):
    data_set = get_baselines(normalize=True)
    X = np.load('geom_feat.npy')
    b, n, _ = X.shape
    if no_feature:
        X = torch.eye(n)
        X = X.reshape((1, n, n))
        X = X.repeat(b, 1, 1)
        print(X.shape)
    DX = []
    for d in data_set["dx_label"]:
        DX.append(int(d) - 1)
    train_fold, test_fold = get_train_test_fold(X, DX)
    return torch.FloatTensor(data_set["adjacency_matrix"]), torch.FloatTensor(X),  \
           torch.tensor(DX), train_fold, test_fold


def accuracy(output, labels):
    total = 0
    for i in range(len(output)):
        preds = output[i].max(1)[1].type_as(labels[i])
        correct = preds.eq(labels[i]).double()
        correct = correct.sum()
        total += len(labels[i])
    return correct / total


def create_geom_feat(adj):
    adj_np = np.array(adj)
    eps = 1e-10
    adj_np[adj_np == 0] = eps
    n = len(adj_np)
    X = np.zeros((n, 2))
    X[:, 0] = bct.betweenness_wei(1/adj_np) / (n - 1) / (n - 2)
    X[:, 1] = bct.clustering_coef_wu(adj_np)
    return X


def show_matrix(adj, recovered):
    from matplotlib import pyplot as plt
    b, n, z = adj.shape
    for i in range(b):
        plt.subplot(2, b, (i + 1))
        plt.imshow(adj[i].cpu().data.numpy())
        plt.subplot(2, b, (i + 1 + b))
        plt.imshow(recovered[i].cpu().data.numpy())
    plt.show()


if __name__ == '__main__':
    data_set = get_baselines()
    X = []
    for i, d in enumerate(data_set["adjacency_matrix"]):
        print("sub: ", i)
        X.append(create_geom_feat(d))

    np.save('geom_feat', np.array(X))
    print(torch.tensor(X).shape)