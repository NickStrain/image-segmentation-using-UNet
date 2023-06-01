import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class double_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(double_conv,self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),)
    def forward(self,x):
        x = self.doubleconv(x)
        return (x)

class unet(nn.Module):
    def __init__(self,in_channel=3,out_channel=2,feature =[64,128,256,512]):
        super(unet,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.ModuleList()
        self.down = nn.ModuleList()
        #downside
        for _ in feature:
           self.down.append(double_conv(in_channel,_))
           in_channel = _
        #upside
        for _ in reversed(feature):
            self.up.append(nn.ConvTranspose2d(_*2,_,kernel_size=2,padding=2))
            self.up.append(double_conv(_*2,_))
        #lower
        self.bottelneck = double_conv(feature[-1],feature[-1]*2)
        self.final_conv = nn.Conv2d(feature[0],out_channel,1)

    def forward(self,x):
        skip_connection  = []
        for dow in self.down:
            x = dow(x)
            skip_connection.append(x)
            x = self.pool(x)
        x = self.bottelneck(x)
        skip_connection = skip_connection[::-1]

        for idx in range(0,len(self.up),2):
            x = self.up[idx](x)
            skip_connection1 = skip_connection[idx//2]

            if x.shape != skip_connection1.shape:
                x = TF.resize(x,size=skip_connection1.shape[2:])
            concat = torch.cat((skip_connection1,x),1)
            x = self.up[idx+1](concat)



        return self.final_conv(x)

sample = torch.randn((3,1,161,161))
model = unet(in_channel=1,out_channel=1)
pre = model(sample)
print(sample.shape)
print(pre.shape)

