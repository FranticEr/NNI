from torch import nn
from typing import *
from torch.nn import init

class Input_Layer(nn.Module):
    def __init__(self, in_channels,out_channels,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=7,stride=1,padding=3)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1=nn.ReLU()
        self.maxp1=nn.MaxPool1d(kernel_size=3,stride=2)
    def __call__(self,x, *args: Any, **kwds: Any) -> Any:
        x=self.conv1(x)
        x=self.bn1(x)
        
        x=self.relu1(x)
        
        x=self.maxp1(x)
        return x 
    def InitalParameter(self,initMathod):
        for m in self.modules():
            pass
        pass
class Down_Sample_Layer(nn.Module):
    def __init__(self, in_channels,out_channels,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv2=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2=nn.ReLU()
    def __call__(self, x,*args: Any, **kwds: Any) -> Any:
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        return x


    
    
class Classfier_Layer(nn.Module):
    def __init__(self, in_features,out_features,hidden_num,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        self.flatten=nn.Flatten()
        
        self.relu1=nn.ReLU()

        self.linear1=nn.Linear(in_features=in_features,out_features=hidden_num)

        self.relu2=nn.ReLU()

        self.linear2=nn.Linear(in_features=hidden_num,out_features=out_features)
    def __call__(self,x, *args: Any, **kwds: Any) -> Any:
        x=self.avgpool(x)
        x=self.flatten(x)
        #print(x.shape)
        x=self.relu1(x)
        #print(x.shape)
        x=self.linear1(x)
        #print(x.shape)
        x=self.relu2(x)
        x=self.linear2(x)

        return x




from typing import Any


class simple_cnn1d(nn.Module):
    def __init__(self, input_channels,num_classes,list_down:list,num_hidden=32,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.list_down=list_down
        #print(self.list_down)
        self.num_classes=num_classes
        #print(num_classes)

        self.input_layer=Input_Layer(in_channels=input_channels,out_channels=list_down[0])
        for i in range(len(self.list_down)-1):
            #print(i)
            setattr(self,f'down_blk{i}',Down_Sample_Layer(self.list_down[i],self.list_down[i+1]))
        self.classfier_layer=Classfier_Layer(self.list_down[-2],self.num_classes,num_hidden)
       
    def __call__(self, x,*args: Any, **kwds: Any) -> Any:
        x=self.input_layer(x)
        #print(self.list_down)
        for i in range(len(self.list_down)-1):
            blk=getattr(self,f"down_blk{i}")
            x=blk(x)
        x=self.classfier_layer(x)
        return x

