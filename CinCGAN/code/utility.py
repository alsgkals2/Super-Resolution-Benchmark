import os
import math
import time
import datetime
from functools import reduce

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import random
def set_multiprosessing():
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

def set_seeds(seed=42 ,isCuda = True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if isCuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # for faster training, but not deterministic

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = '../experiment/' + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            misc.imsave('{}{}.png'.format(filename, p), ndarr)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
 
def calc_rmse(true, pred):
    print(true)
    print(pred)
    score = math.sqrt(np.mean((true-pred)**2))
    return score   

def calc_psnr(true, pred, pixel_max=1):
    np_true = np.array(true.cpu().detach())
    np_pred = np.array(pred.cpu().detach())
    score = 20*np.log10(pixel_max/calc_rmse(np_true, np_pred))
    return score

class PSNR(object):
    def __init__(self, gpu=True, val_max=1, val_min=0, ycbcr=True):
        super(PSNR,self).__init__()
        self.val_max = val_max
        self.val_min = val_min
        self.gpu = gpu
        self.ycbcr = ycbcr
    
    def __call__(self,x,y):
        """
        if x.is_cuda:
            x = x.detach().cpu()
        if y.is_cuda:
            y = y.detach().cpu()
        """
        assert len(x.size()) == len(y.size())
        with torch.no_grad():
            x_lum = rgb_to_ycbcr(x)[:,0]
            y_lum = rgb_to_ycbcr(y)[:,0]
            # if len(x.size()) == 3:
            #     mse = torch.mean((y-x)**2)
            #     psnr = 20*torch.log10(torch.tensor(self.val_max-self.val_min, dtype=torch.float).cuda(self.gpu)) - 10*torch.log10(mse)
            #     return psnr
            # elif len(x.size()) == 4:
        
            mse = torch.mean((y_lum-x_lum)**2, dim=[1,2])
            psnr = 20*torch.log10(torch.tensor(self.val_max-self.val_min, dtype=torch.float).to('cuda')) - 10*torch.log10(mse)
            print("psnr is ---> ",torch.mean(psnr))
            return torch.mean(psnr)

def rgb_to_ycbcr(image):
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)



def calc_psnr_pixsh(sr, hr, scale, rgb_range, benchmark=True):
    psnr = 0
    # diff = (sr - hr).data.div(rgb_range)
    sr.data.div(rgb_range)
    hr.data.div(rgb_range)
    print(sr.shape, hr.shape)
    hr = torch.unsqueeze(hr,0)
    shave = scale + 6
    sr = sr[:, :, shave:-shave, shave:-shave]
    for i in range(2*shave-1):
        for j in range(2*shave-1):
            print(hr[:, :, i:-2*shave+i, j:-2*shave+j].shape)
            print(shave)
            valid = (sr-hr[:, :, i:-2*shave+i, j:-2*shave+j])
            mse = valid.pow(2).mean()
            if psnr < -10 * math.log10(mse):
                psnr = -10* math.log10(mse)


    return psnr
# def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
        
def calc_psnr_pixsh(sr, hr, scale, rgb_range, benchmark=True):
    psnr = 0
    # diff = (sr - hr).data.div(rgb_range)
    sr.data.div(rgb_range)
    hr.data.div(rgb_range)
    print(sr.shape, hr.shape)
    hr = torch.unsqueeze(hr,0)
    shave = scale + 6
    sr = sr[:, :, shave:-shave, shave:-shave]
    for i in range(2*shave-1):
        for j in range(2*shave-1):
            print(hr[:, :, i:-2*shave+i, j:-2*shave+j].shape)
            print(shave)
            valid = (sr-hr[:, :, i:-2*shave+i, j:-2*shave+j])
            mse = valid.pow(2).mean()
            if psnr < -10 * math.log10(mse):
                psnr = -10* math.log10(mse)

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler

