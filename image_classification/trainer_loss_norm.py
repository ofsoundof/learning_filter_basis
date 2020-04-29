import math
import random

import utility
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as tu
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.epoch = 0
        self.optimizer = utility.make_optimizer(args, self.model, ckp=ckp)
        self.scheduler = utility.make_scheduler(
            args,
            self.optimizer,
            resume=len(self.loss.log_test)
        )

        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.model.find('INQ') >= 0:
            self.inq_steps = args.inq_steps
        else:
            self.inq_steps = None

    def train(self):
        # epoch, _ = self.start_epoch()
        self.epoch += 1
        epoch = self.epoch
        self.model.begin(epoch, self.ckp)
        self.loss.start_log()
        if self.args.model == 'DECOMPOSE_COOLING':
            if epoch % 250 == 0 and epoch > 0 and epoch <500:
                self.optimizer = utility.make_optimizer(self.args, self.model, ckp=self.ckp)
                # for group in self.optimizer.param_groups:
                #     group.setdefault('initial_lr', self.args.lr)
                self.scheduler = utility.make_scheduler(
                    self.args,
                    self.optimizer,
                    resume=len(self.loss.log_test)
                )
                self.model.model.reload()
        self.start_epoch()
        timer_data, timer_model = utility.timer(), utility.timer()
        n_samples = 0
        if self.args.loss_norm:
            parent_module = import_module('model.' + self.args.model.split('_')[0].lower())
            current_module = import_module('model.' + self.args.model.lower())
            parent_model = parent_module.make_model(self.args)

        for batch, (img, label) in enumerate(self.loader_train):
            #if batch <=1:
            img, label = self.prepare(img, label)
            n_samples += img.size(0)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            prediction = self.model(img)
            loss, _ = self.loss(prediction, label)
            if self.args.loss_norm:
                loss_norm = current_module.loss_norm_difference(self.model.model, parent_model, self.args, 'L2')
                loss_weight_norm = 0.05 * loss_norm[1]
                loss_weight = loss_norm[0]
                loss = loss_weight + loss_weight_norm + loss
            loss.backward()
            self.optimizer.step()

            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0:
                if self.args.loss_norm:
                    self.ckp.write_log(
                        '{}/{} ({:.0f}%)\t'
                        'NLL: {:.3f}\t'
                        'Top1: {:.2f} / Top5: {:.2f}\t'
                        'Total {:<2.4f}/Diff {:<2.5f}/Norm {:<2.5f}\t'
                        'Time: {:.1f}+{:.1f}s'.format(
                            n_samples,
                            len(self.loader_train.dataset),
                            100.0 * n_samples / len(self.loader_train.dataset),
                            *(self.loss.log_train[-1, :] / n_samples),
                            loss.item(), loss_weight.item(), loss_weight_norm.item(),
                            timer_model.release(),
                            timer_data.release()
                        )
                    )
                else:
                    self.ckp.write_log(
                        '{}/{} ({:.0f}%)\t'
                        'NLL: {:.3f}\t'
                        'Top1: {:.2f} / Top5: {:.2f}\t'
                        'Time: {:.1f}+{:.1f}s'.format(
                            n_samples,
                            len(self.loader_train.dataset),
                            100.0 * n_samples / len(self.loader_train.dataset),
                            *(self.loss.log_train[-1, :] / n_samples),
                            timer_model.release(),
                            timer_data.release()
                        )
                    )

            timer_data.tic()

        self.model.log(self.ckp)
        self.loss.end_log(len(self.loader_train.dataset))

    def test(self):
        # epoch = self.scheduler.last_epoch + 1
        epoch = self.epoch
        self.ckp.write_log('\nEvaluation:')
        self.loss.start_log(train=False)
        self.model.eval()
        timer_test = utility.timer()

        with torch.no_grad():
            for img, label in tqdm(self.loader_test, ncols=80):
                img, label = self.prepare(img, label)
                torch.cuda.synchronize()
                timer_test.tic()
                # a=time.time()
                # from IPython import embed; embed()
                prediction = self.model(img)
                torch.cuda.synchronize()
                timer_test.hold()
                # b = time.time()-a
                # print('The elapse time is {}'.format(b))
                self.loss(prediction, label, train=False)

                if self.args.debug: self._analysis()
        mem = torch.cuda.max_memory_allocated()/1024.0**2
        self.loss.end_log(len(self.loader_test.dataset), train=False)

        # Lower is better
        best = self.loss.log_test.min(0)
        for i, measure in enumerate(('Loss', 'Top1 error', 'Top5 error')):
            self.ckp.write_log(
                '{}: {:.3f} (Best: {:.3f} from epoch {})'.format(
                    measure,
                    self.loss.log_test[-1, i],
                    best[0][i],
                    best[1][i] + 1 if len(self.loss.log_test) == len(self.loss.log_train) else best[1][i]
                )
            )
        total_time = timer_test.release()
        is_best = self.loss.log_test[-1, self.args.top] <= best[0][self.args.top]
        self.ckp.save(self, epoch, is_best=is_best)
        self.ckp.save_results(epoch, self.model)
        self.scheduler.step()

    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half': x = x.half()
            return x

        return [_prepare(a) for a in args]

    def start_epoch(self):
        # epoch = self.scheduler.last_epoch + 1
        # self.epoch += 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2}'.format(self.epoch, lr))

        return self.epoch, lr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            # epoch = self.scheduler.last_epoch + 1
            epoch = self.epoch
            return epoch >= self.args.epochs

    def _analysis(self):
        flops = torch.Tensor([
            getattr(m, 'flops', 0) for m in self.model.modules()
        ])
        flops_conv = torch.Tensor([
            getattr(m, 'flops', 0) for m in self.model.modules() if isinstance(m, nn.Conv2d)
        ])
        flops_ori = torch.Tensor([
            getattr(m, 'flops_original', 0) for m in self.model.modules()
        ])

        print('')
        print('FLOPs: {:.2f} x 10^8'.format(flops.sum() / 10**8))
        print('Compressed: {:.2f} x 10^8 / Others: {:.2f} x 10^8'.format(
            (flops.sum() - flops_conv.sum()) / 10**8 , flops_conv.sum() / 10**8
        ))
        print('Accel - Total original: {:.2f} x 10^8 ({:.2f}x)'.format(
            flops_ori.sum() / 10**8, flops_ori.sum() / flops.sum()
        ))
        print('Accel - 3x3 original: {:.2f} x 10^8 ({:.2f}x)'.format(
            (flops_ori.sum() - flops_conv.sum()) / 10**8,
            (flops_ori.sum() - flops_conv.sum()) / (flops.sum() - flops_conv.sum())
        ))
        input()

