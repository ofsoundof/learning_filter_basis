import os
from data import srdata
import glob


class DIV2KSUB(srdata.SRData):
    def __init__(self, args, train=True):
        super(DIV2KSUB, self).__init__(args, train)
        self.repeat = round(args.test_every / (args.n_train / args.batch_size))
        self.n_train = args.n_train

    def _scan(self):
        list_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*.png')))
        list_lr = [sorted(glob.glob(os.path.join(self.dir_lr + '{}'.format(s), '*.png'))) for s in self.scale]
        #for si, s in enumerate(self.scale):

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/super_resolution/DIV2K'
        self.dir_hr = os.path.join(self.apath, 'GT_sub')
        self.dir_lr = os.path.join(self.apath, 'GT_sub_bicLRx')
        self.ext = '.png'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def __len__(self):
        if self.train:
            return self.n_train * self.repeat #len(self.images_hr) * self.repeat
        else:
            return self.n_train #len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % self.n_train #len(self.images_hr)
        else:
            return idx