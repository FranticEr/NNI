from .Trainer import Trainer
from .operation import *

class ClassfierTrainer(Trainer):
    def __init__(self,net,epochNum,optimizer,lossFunction):
        super().__init__(net,epochNum,optimizer,lossFunction)
        self.SetTrainer(net,epochNum,optimizer,lossFunction)
        pass

    def run(self):
        self.TrainEpochs()
    
    def ForwardBatch(self,net,batchData):
        x=batchData[0]
        y=batchData[1]
        y_hat=net(x)
        return y,y_hat

    def SetTrainer(self,net=None,epochNum=None,optimizer=None,lossFunction=None):
        if not net==None:
            self.Net=net
        if not optimizer==None:
            self.Optimizer=optimizer
        if not lossFunction==None:
            self.LossFuntion=lossFunction
        if not epochNum==None:
            self.EpcochNum=epochNum
                        
    def TrainBatch(self,net,batchData,optimizer,lossFunction):
        self.SetTrainer(net=net,optimizer=optimizer,lossFunction=lossFunction)
        optimizer.zero_grad()
        y,y_hat=self.ForwardBatch(net,batchData)
        loss=lossFunction(y_hat,y)
        loss.backward()
        optimizer.step()
        return y,y_hat,loss
        pass
    def EvalOneEpoch(self,net,dataloader):
        self.SetTrainer(net)
        net.eval()
        epochY=torch.tensor([])
        epochYhat=torch.tensor([])
        with torch.no_grad():
            for i,data in enumerate(dataloader):
                y,y_hat=self.ForwardBatch(net,data)
                epochY=torch.concat([epochY,y],dim=0)
                epochYhat=torch.concat([epochYhat,y_hat],dim=0)
        return epochY,epochYhat
    def TrainOneEpoch(self,net,dataloader,optimizer,lossFunction):
        epochY=torch.tensor([])
        epochLoss=torch.tensor([])
        epochYhat=torch.tensor([])
        for i,data in enumerate(dataloader):
            y,y_hat,loss=self.TrainBatch(net,data,optimizer,lossFunction)
            epochY=torch.concat([epochY,y.detach().cpu()],dim=0)
            epochLoss=torch.concat([epochLoss,torch.tensor([loss.item()])],dim=0)
            epochYhat=torch.concat([epochYhat,y_hat.detach().cpu()],dim=0)
        return epochY,epochYhat,epochLoss
        pass
    def TrainEpochs(self,epochNum,net,traindataloader,testdataloader,optimizer,lossFunction):
        for i in range(epochNum):
            trainEpochY,trainEpochYhat,trainEpochLoss=self.TrainOneEpoch(net,traindataloader,optimizer,lossFunction)
            testEpochY,testEpochYhat=self.EvalOneEpoch(net,testdataloader)
        pass
        pass
    def Statitics(self):
        pass
    def KflodTrain(self):
        pass



    def SetNet(self):
        pass
    def SetEpochNum(self):
        pass
    def SetDataloader(self):
        pass
    def SplitDataloader(self):
        pass
    
    pass

