import os
import cv2
import numpy as np

def makeSample(img_path1,img_path2,sample_path,sample_size=5):
    img1=cv2.imread(img_path1)[...,0]
    img2=cv2.imread(img_path2)[...,0]
    img1=np.pad(img1,2)
    img2=np.pad(img2,2)
    img_size=img1.shape
    for i in range(img_size[0]-sample_size+1):
        for j in range(img_size[1]-sample_size+1):
            img_pic1=img1[i:i+sample_size,j:j+sample_size]
            img_pic2=img2[i:i+sample_size,j:j+sample_size]
            img_pic1=np.expand_dims(img_pic1,-1)
            img_pic2=np.expand_dims(img_pic2,-1)
            sample=np.concatenate([img_pic1,img_pic2],-1)
            sample=np.swapaxes(np.swapaxes(sample,0,2),1,2)

            np.save(sample_path+'{}_{}.npy'.format(i+1,j+1),sample)

def main():
    FarmLandC_img_path1, FarmLandC_img_path2 = './data/dataset/FarmLandC/200806.bmp', './data/dataset/FarmLandC/200906.bmp'
    FarmLandD_img_path1, FarmLandD_img_path2 = './data/dataset/FarmLandD/200806.bmp', './data/dataset/FarmLandD/200906.bmp'
    Ottwa_img_path1, Ottwa_img_path2 = './data/dataset/Ottwa/199707.png', './data/dataset/Ottwa/199708.png'

    FarmLandC_sample_path='../dataset/sample/FarmLandC/'
    FarmLandD_sample_path='../dataset/sample/FarmLandD/'
    Ottwa_sample_path='../dataset/sample/Ottwa/'

    if not os.path.exists(Ottwa_sample_path):
        os.makedirs(Ottwa_sample_path)

    makeSample(Ottwa_img_path1,Ottwa_img_path2,Ottwa_sample_path)

if __name__=='__main__':
    main()