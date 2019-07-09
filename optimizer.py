import torch
import torch.nn.modules.loss


def recon_loss(_A, A, nbin=10):
    hst_raw = torch.histc(A, bins=nbin).float()
    hst = hst_raw/hst_raw.sum()
    #hst = sfm(hst_raw)
    ind = (A * nbin).type(torch.LongTensor)
    cnt = hst[ind]
    loss = (A - _A)**2
    rec_loss = (loss / cnt).sum()
    #print("Reconstruction loss {}".format(rec_loss.item()))
    return rec_loss


def KLD(mu, logvar, n_nodes):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    #print("KDL: {}".format(KLD))
    return KLD