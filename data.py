import os
import nibabel as nib
import cv2
import linecache

imgdirs = os.listdir('../training/')

with open('label.txt', 'a') as f:
    for imgdir in imgdirs:
        print('../training/'+imgdir+'/Inco.cfg')
        line = linecache.getline('../training/'+imgdir+'/Inco.cfg', 3)
        print(line)
        # label = line.strip().split()[1]
        # data = nib.load('../training/'+imgdir+'/'+imgdir+'_4d.nii.gz')
        # imgs = data.get_data()
        # x,y,z,t = imgs.shape
        # if not os.path.exists('../train/'+imgdir):
        #     os.makedirs('../train/'+imgdir)
        # for i in range(t):
        #     for j in range(z):
        #         imgname = '../train/'+imgdir+'/'+str(i)+str(j)+'.png'
        #         cv2.imwrite(imgname, imgs[:,:,j,i])
        #         f.write(imgname+' '+label+'\n')
        #         print(imgname+'done!')