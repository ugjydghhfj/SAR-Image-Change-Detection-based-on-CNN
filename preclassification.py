import cv2
import numpy as np
import os

#计算相似度矩阵
def similityMatrix(img_path1,img_path2):
    img1=cv2.imread(img_path1)[...,0]
    img2=cv2.imread(img_path2)[...,0]
    img_size=img1.shape
    simility=np.zeros(img_size)
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            if int(img1[i][j])+int(img2[i][j])!=0:
                simility[i][j]=np.abs(int(img1[i][j])-int(img2[i][j]))/(int(img1[i][j])+int(img2[i][j]))
            else:
                simility[i][j]=0
    return simility

#空间隶属度矩阵
def spaceMembership(membership_mat,index,v,size):
    BN=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    i=index//size[1]
    j=index%size[1]
    space=0
    for dx,dy in BN:
        nx,ny=i+dx,j+dy
        if (nx>=0 and nx<size[0]) and (ny>=0 and ny<size[1]):
            space+=membership_mat[nx*size[1]+ny][v]
    return space

#SFCM
def SFCM(simility,c_cluster=2,m=2,p=2,q=2,eps=0.0001,epoch=100):
    data=simility.reshape(-1,1)
    membership_mat=np.random.random((len(data),c_cluster))
    membership_mat=np.divide(membership_mat,np.sum(membership_mat,axis=1)[:,np.newaxis])

    for num in range(epoch):
        old_membership_mat=membership_mat
        working_membership_mat=membership_mat**m
        v_i=np.divide(np.dot(working_membership_mat.T,data),np.sum(working_membership_mat.T,axis=1)[:,np.newaxis])

        distence_membership_mat=np.zeros((len(data),c_cluster))
        for i,x in enumerate(data):
            for j,v in enumerate(v_i):
                distence_membership_mat[i][j]=np.linalg.norm(x-v,2)

        for i in range(len(data)):
            for j in range(c_cluster):
                membership_mat[i][j]=1./np.sum((distence_membership_mat[i][j]/distence_membership_mat[i])**(2/(m-1)))

        space_membership_mat=np.zeros((len(data),c_cluster))
        for i in range(len(data)):
            for j in range(c_cluster):
                space_membership_mat[i][j]=spaceMembership(membership_mat,i,j,simility.shape)

        p_membership_mat,q_membership_mat=membership_mat**p,space_membership_mat**q
        for i in range(len(data)):
            for j in range(c_cluster):
                membership_mat[i][j]=p_membership_mat[i][j]*q_membership_mat[i][j]
        membership_mat=np.divide(membership_mat,np.sum(membership_mat,axis=1)[:,np.newaxis])

        if np.sum(abs(membership_mat-old_membership_mat))<eps:
            break

    data=np.argmax(membership_mat,axis=1)
    image=data.reshape(simility.shape)
    return image

#标签可视化
def bilary2pic(image):
    size=image.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if image[i][j]==1:
                image[i][j]=255

    image=np.expand_dims(image,-1)
    image=np.concatenate([image,image,image],axis=-1)
    return image

def main():
    FarmLandC_img_path1,FarmLandC_img_path2='./data/dataset/FarmLandC/200806.bmp','./data/dataset/FarmLandC/200906.bmp'
    FarmLandD_img_path1,FarmLandD_img_path2='./data/dataset/FarmLandD/200806.bmp','./data/dataset/FarmLandD/200906.bmp'
    Ottwa_img_path1,Ottwa_img_path2='./data/dataset/Ottwa/199707.png','./data/dataset/Ottwa/199708.png'

    FarmLandC_preclassification_path='./data/dataset/FarmLandC/'
    FarmLandD_preclassification_path='./data/dataset/FarmLandD/'
    Ottwa_preclassification_path='./data/dataset/Ottwa/'

    FarmLandC_lable_path='./data/lable/FarmLandC/'
    FarmLandD_lable_path='./data/lable/FarmLandD/'
    Ottwa_lable_path='./data/lable/Ottwa/'

    if not os.path.exists(FarmLandD_lable_path):
        os.makedirs(FarmLandD_lable_path)

    simility=similityMatrix(FarmLandD_img_path1,FarmLandD_img_path2)
    image=SFCM(simility)
    cv2.imwrite(FarmLandD_lable_path+'lable.png',image)

    image=bilary2pic(image)
    cv2.imwrite(FarmLandD_preclassification_path+'preclassification.png',image)

if __name__=='__main__':
    main()