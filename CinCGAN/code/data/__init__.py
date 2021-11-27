import os
from dataloader import MSDataLoader, BaseDataLoader
import numpy as np
# import torchvision.transforms as TF
from torchvision import transforms
from importlib import import_module

import torchvision.datasets as datasets
# from dataloader import MSDataLoader
# from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from PIL.Image import Image
import cv2

class Data:
    def __init__(self, args):
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False
        from torch.utils.data import DataLoader
        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            aug = transforms.Compose([
                # transforms.Resize((128,128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ])
            trainset = getattr(module_train, args.data_train)(args,train=True, transform = aug)
            self.loader_train = DataLoader(trainset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0, pin_memory=True)
            # self.loader_train = MSDataLoader(
            #     args,
            #     trainset,
            #     batch_size=16,
            #     shuffle=True,
            #     **kwargs
            # )
            # options = {}
            # options['bs'] = args.batch_size
            # self.loader_train =BaseDataLoader(args, trainset,
            # batch_size = args.batch_size
            # )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
            if not args.benchmark_noise:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:
                module_test = import_module('data.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )

        else:
            module_test = import_module('data.' +  args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)
            self.loader_test = DataLoader(testset, batch_size=1,
                        shuffle=False, num_workers=0)
        # self.loader_test = MSDataLoader(
        #     args,
        #     testset,
        #     batch_size=1,
        #     shuffle=False,
        #     **kwargs
        # )
    
            # trainset = datasets.ImageFolder('/home/mhkim/SuperResolution/DS/DIV2K/DS_CinCGAN/valid', val_aug),
            # options = {}
            # options['bs'] = args.batch_size
            # module_test = import_module('data.' + args.data_train.lower())
            
            # self.loader_test =BaseDataLoader(
            #     inp_dir = '/home/mhkim/SuperResolution/DS/DIV2K/DS_CinCGAN/valid', 
            #     options=options,
            #     args = args,
            #     train = False
                
                # **kwargs
            # )

