
import torch.nn as nn

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type='L2'):
        super(ReconstructionLoss, self).__init__()
        if loss_type == 'L2':
            self.loss_fn = nn.MSELoss()  # 均方误差损失
        elif loss_type == 'L1':
            self.loss_fn = nn.L1Loss()   # 绝对差损失
        else:
            raise ValueError("loss_type should be either 'L2' or 'L1'")

    def forward(self, input_tensor, output_tensor):
        return self.loss_fn(output_tensor, input_tensor)
