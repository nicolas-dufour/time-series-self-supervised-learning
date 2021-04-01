import torch
import torch.nn as nn


class TripletLossLogi(nn.Module):
    '''
    The contrastive Logistic Loss.

    Parameters:
    ----------
        temp: The temperature parameter.
    '''

    def __init__(self, temp=1):
        super().__init__()
        self.temp = temp

    def forward(self, batch_emb, pos_batch_emb):
        batch_size = batch_emb.shape[0]

        batch_emb = batch_emb / batch_emb.norm(dim=-1)[:, None]
        pos_batch_emb = pos_batch_emb / pos_batch_emb.norm(dim=-1)[:, None]

        cov = torch.mm(batch_emb, batch_emb.t().contiguous())
        sim = torch.sigmoid(-cov/ self.temp)

        mask = ~torch.eye(batch_size, device=sim.device).bool()
        neg = sim.masked_select(mask).view(batch_size, -1)
        neg_loss = -torch.sum(torch.log(neg), dim=-1)

        pos = torch.sigmoid(torch.sum(batch_emb * pos_batch_emb, dim=-1)/self.temp)

        pos_loss = -torch.log(pos)

        return (pos_loss + neg_loss).mean()


class TripletLossXent(nn.Module):
     '''
        The contrastive Logistic Loss.

        Parameters:
        ----------
            temp: The temperature parameter.
        '''

    def __init__(self, temp=1):
        super().__init__()
        self.temp = temp

    def forward(self, batch_emb, pos_batch_emb):
        batch_size = batch_emb.shape[0]

        batch_emb = batch_emb / batch_emb.norm(dim=-1)[:, None]
        pos_batch_emb = pos_batch_emb / pos_batch_emb.norm(dim=-1)[:, None]

        cov = torch.mm(batch_emb, batch_emb.t().contiguous())
        sim = torch.exp(cov/self.temp)

        mask = ~torch.eye(batch_size, device=sim.device).bool()
        neg = sim.masked_select(mask).view(batch_size, -1)
        neg_loss = torch.log(torch.sum(neg, dim=-1))

        pos_loss = -torch.sum(batch_emb * pos_batch_emb, dim=-1)/self.temp

        return (pos_loss + neg_loss).mean()