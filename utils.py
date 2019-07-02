from read_file import read_all_subjects
import torch
from helpers import get_train_test_fold


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


def accuracy(output, labels):
    total = 0
    for i in range(len(output)):
        preds = output[i].max(1)[1].type_as(labels[i])
        correct = preds.eq(labels[i]).double()
        correct = correct.sum()
        total += len(labels[i])
    return correct / total
