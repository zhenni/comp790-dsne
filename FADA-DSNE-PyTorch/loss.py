import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# from torch.nn.modules.loss import _Loss, _WeightedLoss

class DSNE_Loss(nn.Module):
    def __init__(self, margin=1.0, fn=False):
        super(DSNE_Loss, self).__init__()
        self.margin = margin
        self.fn = fn
    
    def forward(self, fts, ys, ftt, yt):
        """
        Semantic Alignment Loss
        :param yt: label for the target domain [N]
        :param ftt: features for the target domain [N, K]
        :param ys: label for the source domain [M]
        :param fts: features for the source domain [M, K]
        """
        # ftt -> (N, F) and fts -> (M, F), then
        # distances of pairs will be of shape (N, M, F)
        broadcast_size = (ftt.shape[0], fts.shape[0], ftt.shape[1])
        
        if self.fn:
            # Normalize feature
            fts = fts / torch.norm(fts, dim=1)
            ftt = ftt / torch.norm(ftt, dim=1)
        
        # Compute distances between all fts and ftt pairs of vectors
        fts_rpt = fts.unsqueeze(0).expand(broadcast_size)
        ftt_rpt = ftt.unsqueeze(1).expand(broadcast_size)
        dists = torch.sum((ftt_rpt - fts_rpt) ** 2, dim=2)

        #   - intraclass distances (yt == ys)
        #   - interclass distances (yt != ys)
        broadcast_size_y = broadcast_size[:2]
        ys_rpt = yt.unsqueeze(0).expand(broadcast_size_y)
        yt_rpt = ys.unsqueeze(1).expand(broadcast_size_y)
        
        y_same = torch.eq(yt_rpt, ys_rpt)
        y_diff = torch.logical_not(y_same)
        
        intra_cls_dists = dists * y_same
        inter_cls_dists = dists * y_diff
        
        # Fill 0 values with max to prevent interference with min calculation
        max_dists = torch.max(dists, dim=1, keepdim=True).values
        max_dists = max_dists.expand(broadcast_size_y)
        revised_inter_cls_dists = torch.where(y_same, max_dists, inter_cls_dists)
        
        max_intra_cls_dist = intra_cls_dists.max(dim=1).values
        min_inter_cls_dist = revised_inter_cls_dists.min(dim=1).values
        
        loss = F.relu(max_intra_cls_dist - min_inter_cls_dist + self.margin)
        
        return torch.mean(loss)
        
