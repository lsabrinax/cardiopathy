import os
import nibabel as nib
import cv2
import linecache

imgdirs = os.listdir('../training/')
print('imgdirs length is %d' % len(imgdirs))

with open('label.txt', 'a') as f:
    for imgdir in imgdirs:
        print(imgdir)
        # print('../training/'+imgdir+'/Info.cfg')
        line = linecache.getline('../training/'+imgdir+'/Info.cfg', 3)
        # print(line)
        label = line.strip().split()[1]
        data = nib.load('../training/'+imgdir+'/'+imgdir+'_4d.nii.gz')
        imgs = data.get_data()
        x,y,z,t = imgs.shape
        if not os.path.exists('../train/'+imgdir):
            os.makedirs('../train/'+imgdir)
        for i in range(t):
            for j in range(z):
                imgname = '../train/'+imgdir+'/'+str(i*z+j+1)'.png'
                cv2.imwrite(imgname, imgs[:,:,j,i])
                f.write(imgname+' '+label+'\n')
                print(imgname+'done!')