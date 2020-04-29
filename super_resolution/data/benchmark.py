import os
from data import srdata
import glob


class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = []
        for s in self.scale:
            list_hr.append(sorted(glob.glob(os.path.expanduser(os.path.join(self.dir_hr, '*.png')))))
            list_lr.append(sorted(glob.glob(os.path.expanduser(os.path.join(self.dir_lr, 'X{}/*.png'.format(s))))))

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'super_resolution/benchmark', self.args.data_test)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('', '.png')

