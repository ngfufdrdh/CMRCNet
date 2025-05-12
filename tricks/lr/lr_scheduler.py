from torch.optim.lr_scheduler import CosineAnnealingLR

class WarmupCosineLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, max_lr=1e-4):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
        return lr
