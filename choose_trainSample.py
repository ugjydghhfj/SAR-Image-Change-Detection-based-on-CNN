import os
import cv2
import numpy as np

def chooseSample(img_path1,img_path2,lable_path,des_train_path,des_test_path,n=5,sample_size=5,α=0.6):
    img1=cv2.imread(img_path1)[...,0]
    img2=cv2.imread(img_path2)[...,0]
    lable=cv2.imread(lable_path)[...,0]
    img_size=img1.shape
    for i in range(img_size[0]-n+1):
        for j in range(img_size[1]-n+1):
            lable_pic=lable[i:i+n,j:j+n]
            center_lable=lable_pic[n//2,n//2]
            if np.sum(lable_pic==center_lable)/(n*n)>α:
                img_pic1=img1[i:i+sample_size,j:j+sample_size]
                img_pic2=img2[i:i+sample_size,j:j+sample_size]
                img_pic1=np.expand_dims(img_pic1,-1)
                img_pic2=np.expand_dims(img_pic2,-1)
                feature=np.concatenate([img_pic1,img_pic2],axis=-1)
                feature=np.swapaxes(np.swapaxes(feature,0,2),1,2)
                if i>int(img_size[0]*0.55) and j>int(img_size[1]*0.55):
                    np.save(des_test_path[0]+'{}_{}.npy'.format(i+n//2,j+n//2),feature)
                    np.save(des_test_path[1]+'{}_{}.npy'.format(i+n//2,j+n//2),center_lable)
                else:
                    np.save(des_train_path[0]+'{}_{}.npy'.format(i+n//2,j+n//2),feature)
                    np.save(des_train_path[1]+'{}_{}.npy'.format(i+n//2,j+n//2),center_lable)

def main():
    FarmLandC_img_path1, FarmLandC_img_path2 = './data/dataset/FarmLandC/200806.bmp', './data/dataset/FarmLandC/200906.bmp'
    FarmLandD_img_path1, FarmLandD_img_path2 = './data/dataset/FarmLandD/200806.bmp', './data/dataset/FarmLandD/200906.bmp'
    Ottwa_img_path1, Ottwa_img_path2 = './data/dataset/Ottwa/199707.png', './data/dataset/Ottwa/199708.png'

    FarmLandC_lable_path = './data/lable/FarmLandC/lable.png'
    FarmLandD_lable_path = './data/lable/FarmLandD/lable.png'
    Ottwa_lable_path = './data/lable/Ottwa/lable.png'

    FarmLandC_train_path=['../dataset/train/train/FarmLandC/feature/','../dataset/train/train/FarmLandC/lable/']
    FarmLandC_test_path=['../dataset/train/test/FarmLandC/feature/','../dataset/train/test/FarmLandC/lable/']
    FarmLandD_train_path=['../dataset/train/train/FarmLandD/feature/','../dataset/train/train/FarmLandD/lable/']
    FarmLandD_test_path=['../dataset/train/test/FarmLandD/feature/','../dataset/train/test/FarmLandD/lable/']
    Ottwa_train_path=['../dataset/train/train/Ottwa/feature/','../dataset/train/train/Ottwa/lable/']
    Ottwa_test_path=['../dataset/train/test/Ottwa/feature/','../dataset/train/test/Ottwa/lable/']

    if not os.path.exists(Ottwa_train_path[0]):
        os.makedirs(Ottwa_train_path[0])

    if not os.path.exists(Ottwa_train_path[1]):
        os.makedirs(Ottwa_train_path[1])

    if not os.path.exists(Ottwa_test_path[0]):
        os.makedirs(Ottwa_test_path[0])

    if not os.path.exists(Ottwa_test_path[1]):
        os.makedirs(Ottwa_test_path[1])

    chooseSample(Ottwa_img_path1,Ottwa_img_path2,Ottwa_lable_path,Ottwa_train_path,Ottwa_test_path)

if __name__=='__main__':
    main()