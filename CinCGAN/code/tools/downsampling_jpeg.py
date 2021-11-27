import numpy as np
import os
scale = np.array([2,3,4])
dataset = 'DIV2K'
apath = '../../../../dataset'
quality = 87
hrDir = fullfile(apath,dataset,'DIV2K_train_HR')
lrDir = fullfile(apath,dataset,np.array(['DIV2K_train_LR_bicubic',num2str(quality)]))
if not os.path.exist(str(lrDir)) :
    mkdir(lrDir)

for sc in np.arange(1,len(scale)+1).reshape(-1):
    lrSubDir = fullfile(lrDir,sprintf('X%d',scale(sc)))
    if not os.path.exist(str(lrSubDir)) :
        mkdir(lrSubDir)

os.system('cd .')
hrImgs = dir(fullfile(hrDir,'*.png'))
for idx in np.arange(1,len(hrImgs)+1).reshape(-1):
    imgName = hrImgs(idx).name
    try:
        hrImg = imread(fullfile(hrDir,imgName))
    finally:
        pass
    h,w,__ = hrImg.shape
    
    for sc in np.arange(1,len(scale)+1).reshape(-1):
        ch = int(np.floor(h / scale(sc))) * scale(sc)
        cw = int(np.floor(w / scale(sc))) * scale(sc)
        cropped = hrImg(np.arange(1,ch+1),np.arange(1,cw+1),:)
        lrImg = imresize(cropped,1 / scale(sc),'bicubic')
        __,woExt,ext = os.path.split(imgName)[0],os.path.splitext(os.path.split(imgName)[1])[0],os.path.splitext(os.path.split(imgName)[1])[1]
        lrName = sprintf('%sx%d%s',woExt,scale(sc),'.jpeg')
        imwrite(lrImg,fullfile(lrDir,sprintf('X%d',scale(sc)),lrName),'quality',quality)
    if np.mod(idx,100) == 0:
        print('Processed %d / %d images\n' % (idx,len(hrImgs)))
