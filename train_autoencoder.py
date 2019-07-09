from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import torch
import torch.optim as optim
from model import GCNModelVAE
import torch.nn.functional as F
import time
from utils import load_data_autoencoder, accuracy, show_matrix
from operator import itemgetter
from optimizer import recon_loss, KLD


def train(epoch, train_idx):
    model.train()
    optimizer.zero_grad()
    recovered, mu, logvar = model(features[train_idx], adj[train_idx])
    loss = recon_loss(recovered, adj[train_idx]) +  1e6 * KLD(mu, logvar, n_nodes)
    if epoch%100 == 0:
        print("Loss: ", loss.item())
    loss.backward()
    optimizer.step()
    # print('loss: {}'.format(loss.data))

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
    output, mu, logvar = model(features[idx_test], adj[idx_test])
    mse_loss = torch.nn.MSELoss()
    loss = mse_loss(output, adj[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss.item()))
    show_matrix(adj[idx_test[0:3]], output[idx_test[0:3]])


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden1', type=int, default=2,
                        help='Number of 1st hidden units.')
    parser.add_argument('--hidden2', type=int, default=2, help='Number of 2nd hidden unit')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features, labels, train_idx, test_idx = load_data_autoencoder()

    # Model and optimizer
    feat_dim = 2
    n_nodes = 148
    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout, adj.size(1))
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

    # Train model
    t_total = time.time()
    for i in range(0, len(test_idx)):
        for epoch in range(args.epochs):
            train(epoch, train_idx[i])

        test(test_idx[i])

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


