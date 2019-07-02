from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import torch
import torch.optim as optim
from model import GCN
import torch.nn.functional as F
import time
from utils import load_data, accuracy
from operator import itemgetter

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=148,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, train_idx, test_idx = load_data()

# Model and optimizer
model = GCN(nfeat=148,
            nhid=args.hidden,
            nclass=3,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    for i in range(len(features)):
        features[i] = features[i].cuda()
        adj[i] = adj[i].cuda()
        labels[i] = labels[i].cuda()


def train(epoch, train_idx):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    loss = torch.FloatTensor([0]).cuda()
    out_list = []
    for i in train_idx:
        x = features[i]
        a = adj[i]
        dx = labels[i]
        output = model(x, a)
        out_list.append(output)
        loss.add_(F.nll_loss(output, dx))

    acc_train = accuracy(out_list, itemgetter(*train_idx)(labels))
    if epoch%10==0:
        print("Train Accuracy", acc_train)
    loss.backward()
    optimizer.step()
    #print('loss: {}'.format(loss.data))

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))


def test(idx_test):
    model.eval()
    output = []
    for i in idx_test:
        output.append(model(features[i], adj[i]))

    acc_test = accuracy(output, itemgetter(*idx_test)(labels))
    print("Test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for i in range(0, len(test_idx)):
    for epoch in range(args.epochs):
        train(epoch, train_idx[i])
    test(test_idx[i])
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))