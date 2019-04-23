import os
import nibabel as nib
import cv2
import linecache



imgdirs = os.listdir('../testing/')
print('imgdirs length is %d' % len(imgdirs))

with open('../testing/label.txt', 'a') as f:
    for imgdir in imgdirs:
        print(imgdir)
        # print('../testing/'+imgdir+'/Info.cfg')
        # line = linecache.getline('../testing/'+imgdir+'/Info.cfg', 3)
        # print(line)
        # label = line.strip().split()[1]
        data = nib.load('../testing/'+imgdir+'/'+imgdir+'_frame01.nii.gz')
        imgs = data.get_data()
        x,y,z= imgs.shape
        if not os.path.exists('../test/'+imgdir):
            os.makedirs('../test/'+imgdir)
        # for i in range(t):
            for j in range(z):
                imgname = '../test/'+imgdir+'/'+str(j+1)+'.png'
                cv2.imwrite(imgname, imgs[:,:,j,i])
                f.write(imgname+' '+'\n')
                print(imgname+'done!')




