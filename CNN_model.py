import torch

class SAR_CNN(torch.nn.Module):
    def __init__(self):
        super(SAR_CNN,self).__init__()
        self.cnn=torch.nn.Sequential(
            torch.nn.Conv2d(2,6,kernel_size=2,stride=1,padding=1),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(6,12,kernel_size=2,stride=1,padding=0),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(12,2,kernel_size=1,stride=1,padding=0)
        )

    def forward(self,x):
        x=self.cnn(x)
        x=x.squeeze()
        return x