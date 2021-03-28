import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.num_negative_samples = num_negative_samples

    def forward(self, batch_emb, pos_batch_emb):
        batch_size = batch_emb.shape[0]
        
        batch_emb = batch_emb/batch_emb.norm(dim=-1)[:, None]
        pos_batch_emb = pos_batch_emb/pos_batch_emb.norm(dim=-1)[:, None]
        
        cov = torch.mm(batch_emb, batch_emb.t().contiguous())
        sim = torch.sigmoid(-cov)

        mask  = ~torch.eye(batch_size, device = sim.device).bool()
        neg = sim.masked_select(mask).view(batch_size,-1)
        neg_loss = -torch.sum(torch.log(neg), dim=-1)

        pos = torch.sigmoid(torch.sum(batch_emb*pos_batch_emb, dim=-1))

        pos_loss = -torch.log(pos)

        return (pos_loss + neg_loss).mean()


