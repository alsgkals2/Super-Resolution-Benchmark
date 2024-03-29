import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.utils as utils
import numpy as np
import data
import model
from model.edsr import EDSR
from option import args
import copy
from srresnet import _NetG_DOWN, _NetD
from loss.tvloss import TVLoss
from utility import calc_psnr_pixsh, set_seeds, PSNR, calc_psnr
################temp
from importlib import import_module

import torch
import torch.nn.functional as F
import tqdm

opt = args
# opt.gpus = opt.n_GPUs
opt.start_epoch = 0

# print(opt)

opt.cuda = not opt.cpu
print(f"cuda : {opt.cuda}")
criterion1 = nn.L1Loss()
criterion = nn.MSELoss()
# criterion_ = nn.MSELoss(size_average=False)
criterion_ = nn.MSELoss(reduction='sum')
criterionD = nn.MSELoss()
tvloss = TVLoss()

# torch.set_num_threads(4)

def main():
    # CUDA_LAUNCH_BLOCKING=1
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    global opt, model
    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = 42
    print("===> Seed: ", opt.seed)
    set_seeds(opt.seed,isCuda=opt.cuda)
    cudnn.benchmark = True
    scale = int(args.scale[0])
    print("===> Loading datasets")
    
    opt.n_train = 400
    loader = data.Data(opt)
    opt_high = copy.deepcopy(opt)
    opt_high.offset_train = 400
    opt_high.n_train = 400

    loader_high = data.Data(opt_high)

    training_data_loader = loader.loader_train
    training_high_loader = loader_high.loader_train
    test_data_loader = loader.loader_test

    print("===> Building model")

    GLR = _NetG_DOWN(stride=2)
    GHR = EDSR(args)
    GDN = _NetG_DOWN(stride=1)
    DLR =_NetD(stride=1)
    DHR = _NetD(stride=2)
    GNO = _NetG_DOWN(stride=1)

    Loaded = torch.load('../experiment/model/EDSR_x{}.pt'.format(scale))
    GHR.load_state_dict(Loaded)     
    
    model = nn.ModuleList()
    GDN = nn.DataParallel(GDN)#_NetG_DOWN(stride=1) #generator
    GHR = nn.DataParallel(GHR)#EDSR(args) #얘가 논문 architecture에서 SR임
    GLR = nn.DataParallel(GLR)#_NetG_DOWN(stride=2) #generator
    DLR = nn.DataParallel(DLR)#_NetD(stride=1) #descriminator
    DHR = nn.DataParallel(DHR)#_NetD(stride=2) #descriminator
    GNO = nn.DataParallel(GNO)#_NetG_DOWN(stride=1) #generator
    model.append(GDN) #G1
    model.append(GHR) #SR
    model.append(GLR) #G3
    model.append(DLR) #D1
    model.append(DHR) #D2
    model.append(GNO) #G2
    
    cudnn.benchmark = True
    model = model.to('cuda')
    
    # if opt.cuda:
    #     print("===> Setting GPU")
    #     model = model.to('cuda')
    #     if args.n_GPUs > 1:
    #         model = nn.DataParallel(model)
    
    optG = torch.optim.Adam(list(model[0].parameters())+list(model[1].parameters())+ list(model[2].parameters())+list(model[5].parameters()), lr=opt.lr, weight_decay=0)
    optD = torch.optim.Adam(list(model[3].parameters())+list(model[4].parameters()), lr=opt.lr, weight_decay=0)
    
    #optionally resume from a checkpoint
    if opt.resume:
        saved_model = os.path.join('./checkpoint',opt.resume)
        if os.path.exists(saved_model):
            print("=> loading checkpoint '{}'".format(saved_model))
            checkpoint = torch.load(saved_model)
            opt.start_epoch = checkpoint["epoch"] + 1
            print(f'start_epoch : {opt.start_epoch}')
            checkpoint_optim = checkpoint['optimizer']
            optG.load_state_dict(checkpoint_optim[0])
            optD.load_state_dict(checkpoint_optim[1])
            model.load_state_dict(checkpoint["model"].state_dict())
        elif os.path.isfile('backup.pt'):
            print("=> no checkpoint found at '{}', Being loading backup.pt file...".format(saved_model))
            checkpoint = torch.load('backup.pt')
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint)
        else :
            print("=> no checkpoint found anything, Check the file path of saved model's")
            return -1
        
    step = 2 if opt.start_epoch > opt.epochs else 1

    optimizer =  [optG, optD]

    # print("===> Setting Optimizer")
    if opt.test_only:
        print('===> Testing')
        test(test_data_loader, model ,opt.start_epoch)
        return

    if step == 1:
        print("===> Training Step 1.")
        for epoch in range(opt.start_epoch, opt.epochs + 1):
            train(training_data_loader, training_high_loader, model, optimizer, epoch, False)
            save_checkpoint(model, optimizer, epoch, scale)
            test(test_data_loader, model, epoch)
        torch.save(model.state_dict(),'backup.pt')
    elif step == 2:
        print("===> Training Step 2.")
        opt.lr = 1e-4
        for epoch in range(opt.start_epoch + 1, opt.epochs*2 + 1):
            train(training_data_loader, training_high_loader, model, optimizer, epoch, True)
            save_checkpoint(model, optimizer, epoch, scale)
            test(test_data_loader, model, epoch)

def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (opt.gamma ** ((epoch) // opt.lr_decay))
    return lr

global psnr_best
psnr_best =0.0
def train(training_data_loader, training_high_loader, model, optimizer, epoch, joint=False):
    global psnr_best
    TESTMODE=False
    image, image_ = None,None
    list_D_loss, list_g1_loss = [],[]
    list_psnr_hr = []
    val_learning_rate = epoch-opt.epochs if joint else epoch
    lr = adjust_learning_rate(val_learning_rate)
    
    optG, optD = optimizer

    for param_group in optG.param_groups:
        param_group["lr"] = lr
    for param_group in optD.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, lr))

    model.train()
    for iteration, (batch0, batch1) in tqdm.tqdm(enumerate(zip(training_data_loader, training_high_loader))):
        input_v, target_v = batch0[0], batch0[1] # input domain datase
        input, target, bicubic = batch1[0], batch1[1], batch1[2] # target domain dataset
        # we do not know target_v and input (unsupervised, unpair setting)
        # input : unknown input, target : high resolution image, bicubic : low resolution image downsampled by bicubic kernel
        y_real = torch.ones(input_v.size(0),1,1,1)
        y_fake = torch.zeros(input_v.size(0),1,1,1)
        # print("--------y_real-----------")
        if opt.cuda:
            target = target.to('cuda') /args.rgb_range
            input = input.to('cuda')/args.rgb_range
            bicubic= bicubic.to('cuda')/args.rgb_range
            target_v = target_v.to('cuda')/args.rgb_range
            input_v = input_v.to('cuda')/args.rgb_range
            y_real = y_real.to('cuda')
            y_fake = y_fake.to('cuda')
        if TESTMODE:
            print("--------------------target--------------------")
            print(target)
            print("--------------------input--------------------")
            print(input)
            print("--------------------bicubic--------------------")
            print(bicubic)
        optG.zero_grad()

                # Discriminator loss
        ########### D lr ##############
        optD.zero_grad()

        dn_ = model[0](input_v)
        real_lr = model[3](bicubic)
        fake_lr = model[3](dn_.detach())

        D_lr_loss = \
                        criterionD(fake_lr, y_fake.expand_as(fake_lr)) \
                        + criterionD(real_lr, y_real.expand_as(real_lr))

        (D_lr_loss).backward(retain_graph=True)
        # (D_lr_loss).backward()
        optD.step()
        
        fake_lr_g = model[3](dn_)
        DG_lr_loss = \
                        criterionD(fake_lr_g, y_real.expand_as(fake_lr_g))

        DG_lr_loss.backward(retain_graph=True)
        # DG_lr_loss.backward()
        # D_loss = D_lr_loss+DG_lr_loss
        # D_loss.backward()
        # list_D_loss.append(D_loss.item())
        if TESTMODE:
            print('D_loss : ',D_loss.item())
        # optD.step()

        # TV lr loss
        dn_ = model[0](input_v)    
        if TESTMODE:
            print("dn_dn_dn_dn_dn_dn_1111111111111111")
            print(dn_.mean())
        TV_loss_lr = 0.5 * tvloss(dn_)
        TV_loss_lr.backward(retain_graph=True)

                # Update Generator
        ######### cycle & idt lr ###########

        #idt loss
        bi_l = model[0](bicubic)
        idt_loss_l = \
                     criterion1(bi_l, bicubic)
        idt_loss_l = idt_loss_l * 5
        idt_loss_l.backward(retain_graph=True)

        #cycle loss
        dn_ = model[0](input_v)
        no_ = model[5](dn_) #G2
        
        cyc_loss_l = criterion(no_, input_v) 
        cyc_loss_l = cyc_loss_l * 10
        cyc_loss_l.backward(retain_graph=True)

        # update G(lr)
        optG.step()


        
        # g1_forward_loss = G_lr_loss + cyc_loss_l + idt_loss_l + TV_loss_lr
        # g1_backward_loss = G_lr_loss + cyc_loss_l + idt_loss_l
        # g1_loss = g1_forward_loss + g1_backward_loss
        if TESTMODE:
            print('g1_loss : ',g1_loss.item())
        # g1_loss.backward(retain_graph=True)
        # list_g1_loss.append(g1_loss.item())

        if joint:
        ##########Step 2(Joint training part)##########
        ########### D hr ##############
            optG.zero_grad()
            optD.zero_grad()

            dn_ = model[0](input_v)
            hr_ = model[1](dn_)
            if TESTMODE:
                print("dn_dn_dn_dn_dn_dn_22222222222222")
                print(dn_.mean())
                print("hr_hr_hr_hr_hr_hr_")
                print(hr_.mean())
            real_hr = model[4](target)
            fake_hr = model[4](hr_.detach())
            D_hr_loss = \
                            (criterionD(fake_hr, y_fake.expand_as(fake_hr))\
                             + criterionD(real_hr, y_real.expand_as(real_hr)) )
            if TESTMODE:
                print("TESTMODE1 : ",criterionD(fake_hr, y_fake.expand_as(fake_hr)))
                print("TESTMODE2 : ",criterionD(real_hr, y_real.expand_as(real_hr)))
                print('D_hr_loss : ', D_hr_loss.item())
            (D_hr_loss).backward(retain_graph=True)
            optD.step()
            
            fake_hr_g = model[4](hr_)
            DG_hr_loss = \
                            criterionD(fake_hr_g, y_real.expand_as(fake_hr_g))
            if TESTMODE:
                print("TESTMODE3 : ",criterionD(fake_hr_g, y_real.expand_as(fake_hr_g)))
                print('DG_hr_loss : ', DG_hr_loss.item())

            DG_hr_loss.backward(retain_graph=True)
            # DG_hr_loss.backward()
            #이까진 값 잘뜸
            # TV hr loss
            if TESTMODE:
                print("Start TV HR LOSS IN STEP2--------")
            dn_ = model[0](input_v)    
            hr_ = model[1](dn_) #애초부터 hr_값이 너무 낮아서 나타나는 문제같음
            TV_loss_hr = 2 * tvloss(hr_) #여기가 문젠데
            TV_loss_hr.backward(retain_graph=True)
            # TV_loss_hr.backward()
            if TESTMODE:
                print('TV_loss_hr : ', TV_loss_hr.item())
            #여기로스부터 값이 이상함

            ########## cycle & idt hr ###########        
            bi_ = model[1](bicubic)

            idt_loss = \
                        criterion_(bi_, target)
            if TESTMODE:
                print("TESTMODE3")
                print(f"bi_:{bi_}, target:{target}")
                print(criterion_(bi_, target))

            idt_loss = idt_loss * 5
            idt_loss.backward(retain_graph=True)
            # idt_loss.backward()
            
            dn_ = model[0](input_v) #G1
            hr_ = model[1](dn_) #SR
            lr_ = model[2](hr_) #G3
            cyc_loss = \
                       criterion(lr_, input_v) 
            if TESTMODE:
                print(criterion(lr_, input_v)   )
            cyc_loss = cyc_loss * 10
            
            cyc_loss.backward()
            # cyc_loss.backward()
            # joint_loss = D_hr_loss + DG_hr_loss + TV_loss_hr + idt_loss + cyc_loss
            # joint_loss.backward(retain_graph =True)
            # update G(hr)
            optG.step()
        if iteration % 50 == 0:
            model.eval()
            with torch.no_grad():
                sr_ = model[1](model[0](input)) #GDN (_NetG_DOWN) -> GHR (EDSR)e
                sr_r = model[2](sr_)
                sr = model[2](target) 
                srr = model[1](model[0](sr)) #GLR (_NetG_DOWN,stride=2)->GDN->GHR
                #밑에 메소드랑 똑같이 나오는거 확인함
                psnr_ = -20 *((sr_ - target).pow(2).mean().pow(0.5)).log10()
                psnr = -20*((sr - input).pow(2).mean().pow(0.5)).log10()
            model.train()
            # image = torch.cat([target, sr_, model[1](bicubic)], -2)
            # image_ = torch.cat([input, bicubic, sr, sr_r, model[0](input), model[5](model[0](input))], -2)
            # utils.save_image(image, 'hr_result.png')
            # utils.save_image(image_, 'lr_result.png')
            # utils.save_image(image, f'hr_result_{epoch}e.png') #출력이미지는 batchsize가 16이면 16개 뭉치로 3개나종ㅁ
            # utils.save_image(image_, f'lr_result{epoch}e.png') #16개 뭉치로 6개 나옴
            # utils.save_image(bicubic, f'bicubic_temp_{epoch}e.png')
            # utils.save_image(target, f'target_temp_{epoch}e.png')
            #nan: idt_loss, cyc_loss, D_hr_loss, DG_hr_loss
            # print("===> Epoch[{}]({}/{}): D_loss: {:.6f} g1_loss: {:.6f} | psnr_hr: {:.6f}, psnr_lr {:.6f} "\
            #     .format(epoch, iteration, len(training_data_loader),\
            #         np.average(list_D_loss), np.average(list_g1_loss), psnr_, psnr))
            
            # print("===> Epoch[{}]({}/{}): Loss: idt {:.6f} {:.6f} | cyc {:.6f}  {:.6f} | D {:.6f} {:.6f}, G: {:.6f} {:.6f}, psnr_hr: {:.6f}, psnr_lr {:.6f} "\
            #         .format(epoch, iteration, len(training_data_loader), idt_loss.item(), idt_loss_l.item(), cyc_loss.item(), cyc_loss_l.item(),\
            #             D_hr_loss.item(), D_lr_loss.item(), DG_hr_loss.item(), DG_lr_loss.item(), psnr_, psnr,))
            with open('../experiment/train.txt', 'a') as f:
                if joint:
                    f.write("===> Epoch[{}]({}/{}): Loss: idt {:.6f} {:.6f} | cyc {:.6f} {:.6f} | D {:.6f} {:.6f}, G: {:.6f} {:.6f} | psnr_hr: {:.6f}, psnr_lr {:.6f} "\
                        .format(epoch, iteration, len(training_data_loader), idt_loss.item(), idt_loss_l.item(), cyc_loss.item(), cyc_loss_l.item(),\
                            D_hr_loss.item(), D_lr_loss.item(), DG_hr_loss.item(), DG_lr_loss.item(), psnr_, psnr,)+' \n')
                    
                else:
                    f.write("===> Epoch[{}]({}/{}): Loss: idt {:.6f} | cyc {:.6f} | D {:.6f} , G: {:.6f} | psnr_hr: {:.6f}, psnr_lr {:.6f} "\
                        .format(epoch, iteration, len(training_data_loader), idt_loss_l.item(), cyc_loss_l.item(),\
                            D_lr_loss.item(), DG_lr_loss.item(), psnr_, psnr)+' \n')
                    print("===> Epoch[{}]({}/{}): Loss: idt {:.6f} | cyc {:.6f} | D {:.6f} , G: {:.6f} | psnr_hr: {:.6f}, psnr_lr {:.6f} "\
                        .format(epoch, iteration, len(training_data_loader), idt_loss_l.item(), cyc_loss_l.item(),\
                            D_lr_loss.item(), DG_lr_loss.item(), psnr_, psnr))

         
def test(test_data_loader, model, epoch):
    avg_ = 0
    avg = 0
    n = len(test_data_loader)
    model.eval()
    for iteration, batch in enumerate(test_data_loader):
        input, target, bicubic = batch[0], batch[1], batch[2]
        if opt.cuda: 
            target = target.to('cuda')/args.rgb_range
            input = input.to('cuda')/args.rgb_range
            bicubic = bicubic.to('cuda')/args.rgb_range
            # print(input.shape, target.shape, bicubic.shape) #torch.Size([1, 3, 351, 510]) torch.Size([1, 3, 1356, 2040]) torch.Size([1, 3, 339, 510])

        with torch.no_grad():
            # input = torch.unsqueeze(input,0)
            sr_ = model[1](model[0](input)) #hr
            sr = model[0](input) #lr
            utils.save_image(sr_, 'result/h_{}.png'.format(iteration))
            utils.save_image(sr, 'result/l_{}.png'.format(iteration))
            utils.save_image(bicubic, 'result/bicubic_{}.png'.format(iteration))
            utils.save_image(target, 'result/target_{}.png'.format(iteration))
            # psnr_mh = calc_psnr(sr_, target)

            # print(sr_.shape, target.shape)
            psnr_ = -20 *((sr_ - target).pow(2).mean().pow(0.5)).log10()
            psnr = -20*((sr - input).pow(2).mean().pow(0.5)).log10()
            # psnr_ = calc_psnr_pixsh(sr_, target, args.scale[0], 1) #모서리를 빼고 계산한다는 의미같은데 트릭같음..
            # psnr = calc_psnr_pix  sh(sr, bicubic, args.scale[0], 1)
        avg += psnr#.data
        avg_ += psnr_#.data

        
        print("===> ({}/{}): psnr lr: {:.10f} hr: {:.10f} "\
                    .format(iteration, len(test_data_loader), psnr, psnr_,))
    print('lr psnr', avg/n, 'hr psnr', avg_/n)
    with open('../experiment/test.txt', 'a') as f:
        f.write('{} {} {} {} {}\n'.format(epoch,'lr', avg/n,'hr', avg_/n))

def to_numpy(var):
    return var.data.cpu().numpy()

def save_checkpoint(model, optimizer, epoch, scale=2):
    model_title =  f"model_total_x{scale}_e{epoch}.pth"
    save_path = "../experiment/checkpoint/"
    state = {"epoch": epoch ,"model": model, 'optimizer': [optimizer[0].state_dict(), optimizer[1].state_dict()]}
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(state, save_path + model_title)

    print("Checkpoint saved to {}".format(model_title))

if __name__ == "__main__":
    main()
