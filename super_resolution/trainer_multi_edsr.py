import os
import math
from decimal import Decimal
import torch.nn as nn
import utility
from importlib import import_module

import torch
from tqdm import tqdm

class Trainer(object):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        if self.args.loss_norm:
            pretrain_state = torch.load(self.args.pre_train_optim)
        model_module = import_module('model.' + self.args.model.lower())
        idx_scale = self.args.scale
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            # from IPython import embed; embed(); exit()
            sr = self.model(idx_scale, lr)
            loss = self.loss(sr, hr)
            if self.args.loss_norm:
                loss_norm = model_module.loss_norm_difference(self.model.model, self.args, pretrain_state, 'L2')
                loss_weight_norm = 1.0 * loss_norm[1]
                loss_weight = loss_norm[0]
                loss = loss_weight + loss_weight_norm + loss
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                if self.args.loss_norm:
                    print('Skip this batch {}! (Loss: {}, Loss weight: {}, Loss norm: {}) (Threshold: {})'.
                          format(batch + 1, loss.item(), loss_weight.item(), loss_weight_norm.item(),
                                 self.args.skip_threshold * self.error_last))
                else:
                    print('Skip this batch {}! (Loss: {}) (Threshold: {})'.
                          format(batch + 1, loss.item(), self.args.skip_threshold * self.error_last))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                if self.args.loss_norm:
                    self.ckp.write_log('[{}/{}]\t{}\t{:.3f}+{:.3f}s\tTotal {:<2.4f}/Diff {:<2.5f}/Norm {:<2.5f}'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        self.loss.display_loss(batch),
                        timer_model.release(),
                        timer_data.release(), loss.item(), loss_weight.item(), loss_weight_norm.item()))
                else:
                    self.ckp.write_log('[{}/{}]\t{}\t{:.3f}+{:.3f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        self.loss.display_loss(batch),
                        timer_model.release(),
                        timer_data.release()))
            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):

                    # from IPython import embed; embed();
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    torch.cuda.synchronize()
                    timer_test.tic()
                    sr = self.model(idx_scale, lr)
                    torch.cuda.synchronize()
                    timer_test.hold()
                    # from IPython import embed; embed();
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            div2k=self.args.data_test == 'DIV2K'
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)
                mem = torch.cuda.max_memory_allocated()/1024.0**2
                # from IPython import embed; embed();
                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )
        total_time = timer_test.release()
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
        else:
            torch.save(self.model.get_model().state_dict(),
                os.path.join(self.ckp.dir, '{}_X{}_L{}.pt'.
                             format(self.args.model, self.args.scale[0], self.args.n_resblocks))
            )
        self.scheduler.step()

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

    def loss_weight(self, weights_train, para_loss_type):
        para = torch.load('/scratch_net/ofsoundof/yawli/conadp/softplus_SRBASIS_X3_L20B16P192F256/model/model_latest.pt')
        keys = [k for k, _ in para.items()]
        weights_tar = [para[keys[2]], para[keys[5]], para[keys[8]]]
        loss = 0
        loss_fun = self.loss_type(para_loss_type)
        for i in range(3):
            loss += loss_fun(weights_tar[i], weights_train[i])
            # from IPython import embed; embed();
        return loss

    def loss_type(self, loss_para_type):
        if loss_para_type == 'L1':
            loss_fun = nn.L1Loss()
        elif loss_para_type == 'L2':
            loss_fun = nn.MSELoss()
        else:
            raise NotImplementedError
        return loss_fun


