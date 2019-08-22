import torch

import utility
from data import Data
from model import Model
from loss import Loss
from trainer_loss_norm import Trainer
from option import args

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

#print('Flag')
if checkpoint.ok:
    loader = Data(args)
    model = Model(args, checkpoint)
    loss = Loss(args, checkpoint)
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        if t.scheduler.last_epoch == -1 and not args.test_only:
            t.test()
        t.train()
        t.test()

    checkpoint.done()
