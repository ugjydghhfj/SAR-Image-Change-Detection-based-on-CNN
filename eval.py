import os
import preclassification
import numpy as np
import torch
from CNN_model import SAR_CNN
import cv2

def eval(model_path,sample_path,img_size,result_path):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=SAR_CNN()
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    imageNameList=os.listdir(sample_path)
    image=np.zeros(img_size)
    for num in range(len(imageNameList)):
        imageName=imageNameList[num]
        x=int(imageName.split('_')[0])-1
        y=int(imageName.split('_')[1].split('.')[0])-1
        input=np.load(sample_path+imageName)
        input=torch.from_numpy(input).type(torch.FloatTensor)
        input=input.to(device)
        with torch.no_grad():
            model.eval()
            out=model(input)
        _,index=torch.max(out,dim=-1)
        image[x][y]=index

    image=preclassification.bilary2pic(image)
    cv2.imwrite(result_path,image)

def main():
    model_dict_path = './data/model_dict/'
    FarmLandC_model_dict = model_dict_path + 'FarmLandC_model.pth'
    FarmLandD_model_dict = model_dict_path + 'FarmLandD_model.pth'
    Ottwa_model_dict = model_dict_path + 'Ottwa_model.pth'

    FarmLandC_sample_path = '../dataset/sample/FarmLandC/'
    FarmLandD_sample_path = '../dataset/sample/FarmLandD/'
    Ottwa_sample_path = '../dataset/sample/Ottwa/'

    FarmLandC_size=cv2.imread('./data/dataset/FarmLandC/200806.bmp')[...,0].shape
    FarmLandD_size=cv2.imread('./data/dataset/FarmLandD/200806.bmp')[...,0].shape
    Ottwa_size=cv2.imread('./data/dataset/Ottwa/199707.png')[...,0].shape

    FarmLandC_result='./data/dataset/FarmLandC/result.png'
    FarmLandD_result='./data/dataset/FarmLandD/result.png'
    Ottwa_result='./data/dataset/Ottwa/result.png'

    eval(FarmLandC_model_dict,FarmLandC_sample_path,FarmLandC_size,FarmLandC_result)

if __name__=='__main__':
    main()