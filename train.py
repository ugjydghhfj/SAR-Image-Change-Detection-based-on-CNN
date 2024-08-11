import os.path

import torch
from torch.utils.data import DataLoader
from dataLoader import Data
from CNN_model import SAR_CNN
from torch.utils.tensorboard import SummaryWriter

def train(train_path,test_path,tensorboard_logs,model_dict,epoch=5,batch_size=64):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #数据集
    train_data=Data(train_path[0],train_path[1])
    test_data=Data(test_path[0],test_path[1])

    train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

    #模型
    model=SAR_CNN()
    model=model.to(device)

    #损失与优化
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0003)

    #训练
    writer=SummaryWriter(tensorboard_logs)
    train_total_step=0
    for num in range(epoch):
        print('第{}次训练'.format(num+1))

        model.train()
        for index,(image,lable) in enumerate(train_loader):
            image,lable=image.to(device),lable.to(device)
            out=model(image)
            loss=criterion(out,lable)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_step+=1

            if (index+1)%100==0:
                print('训练次数:{},Loss:{}'.format(index+1,loss.item()))
                writer.add_scalar('train',loss.item(),train_total_step)

        model.eval()
        accuracy=0
        test_total=0
        test_loss=0
        with torch.no_grad():
            for index,(image,lable) in enumerate(test_loader):
                image,lable=image.to(device),lable.to(device)
                out=model(image)
                loss=criterion(out,lable)
                test_loss+=loss.item()
                _,pic=torch.max(out,dim=-1)
                accuracy+=(pic==lable).sum().item()
                test_total+=image.size(0)
        print('loss of test:{}'.format(test_loss))
        print('accuracy of test:{}'.format(accuracy/test_total))
        writer.add_scalar('accuracy',accuracy/test_total,num+1)

    writer.close()
    torch.save(model.state_dict(),model_dict)
    print('模型已保存')



if __name__=='__main__':
    FarmLangC_train_path=['../dataset/train/train/FarmLandC/feature/', '../dataset/train/train/FarmLandC/lable/']
    FarmLangC_test_path=['../dataset/train/test/FarmLandC/feature/', '../dataset/train/test/FarmLandC/lable/']
    FarmLangD_train_path=['../dataset/train/train/FarmLandD/feature/', '../dataset/train/train/FarmLandD/lable/']
    FarmLangD_test_path=['../dataset/train/test/FarmLandD/feature/', '../dataset/train/test/FarmLandD/lable/']
    Ottwa_train_path=['../dataset/train/train/Ottwa/feature/', '../dataset/train/train/Ottwa/lable/']
    Ottwa_test_path=['../dataset/train/test/Ottwa/feature/', '../dataset/train/test/Ottwa/lable/']

    FarmLandC_tensorboard_logs='./data/logs/FarmLandC_logs'
    FarmLandD_tensorboard_logs='./data/logs/FarmLandD_logs'
    Ottwa_tensorboard_logs='./data/logs/Ottwa'

    model_dict_path='./data/model_dict/'
    FarmLandC_model_dict=model_dict_path+'FarmLandC_model.pth'
    FarmLandD_model_dict =model_dict_path+'FarmLandD_model.pth'
    Ottwa_model_dict =model_dict_path+'Ottwa_model.pth'

    if not os.path.exists(model_dict_path):
        os.makedirs(model_dict_path)

    train(Ottwa_train_path,Ottwa_test_path,Ottwa_tensorboard_logs,Ottwa_model_dict)



