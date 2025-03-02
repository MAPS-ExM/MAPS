from torch import optim


class ScheduledOptimizer(optim.Optimizer):
    def __init__(self, optimizer, n_warmup_steps, base_lr):
        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.base_lr = base_lr

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        return getattr(self.optimizer, attr)
    
    def step(self):
        self.n_current_steps += 1
        if self.n_current_steps <= self.n_warmup_steps:
            self._update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _update_learning_rate(self):
        lr = min(self.n_current_steps / self.n_warmup_steps, 1) * self.base_lr 
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def __repr__(self):
        return self.optimizer.__repr__()
    
    

def build_optimizer(model, args):
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.l2_decay,
                              nesterov=True)
    elif args.optimizer.lower() == 'adam':
        if getattr(args, 'l2_decay_drug', None) is not None:
            optimizer = optim.Adam([
                                    {'params': [para for name, para in model.named_parameters() if 'mlp_cat' not in name],
                                     'weight_decay':args.l2_decay,
                                     }, 
                                    {'params': [para for name, para in model.named_parameters() if 'mlp_cat' in name],
                                     'weight_decay':args.l2_decay_drug,
                                     }, 
                                ],
                                betas=(0.9, 0.999),
                                lr=args.lr) 
        else:
            optimizer = optim.Adam(model.parameters(),
                                weight_decay=args.l2_decay,
                                betas=(0.9, 0.999),
                                lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        if getattr(args, 'l2_decay_drug', None) is not None:
            optimizer = optim.AdamW([
                                    {'params': [para for name, para in model.named_parameters() if 'mlp_cat' not in name],
                                     'weight_decay':args.l2_decay,
                                     }, 
                                    {'params': [para for name, para in model.named_parameters() if 'mlp_cat' in name],
                                     'weight_decay':args.l2_decay_drug,
                                     }, 
                                ],
                                betas=(0.9, 0.999),
                                lr=args.lr) 
        else:
            optimizer = optim.AdamW(model.parameters(),
                                weight_decay=args.l2_decay,
                                betas=(0.9, 0.999),
                                lr=args.lr)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented yet")
    
    if getattr(args, 'n_warmup_steps', None) is not None:
        optimizer = ScheduledOptimizer(optimizer, args.n_warmup_steps, args.lr)
        
    return optimizer
