import torch
def trainBatch(net,batchData,optimizer,lossFunction):
    net.train()
    optimizer.zero_grad()
    y,y_hat=forwardBatch(net,batchData)
    loss=lossFunction(y_hat,y)
    loss.backward()
    optimizer.step()
    return y,y_hat,loss

def forwardBatch(net,batchData):
    x=batchData[0]
    y=batchData[1]
    y_hat=net(x)
    return y,y_hat


def trainOneEpoch(net,dataloader,optimizer,lossFunction):
    epochY=torch.tensor([])
    epochLoss=torch.tensor([])
    epochYhat=torch.tensor([])
    for i,data in enumerate(dataloader):
        y,y_hat,loss=trainBatch(net,data,optimizer,lossFunction)
        

        epochY=torch.concat([epochY,y.detach().cpu()],dim=0)
        epochLoss=torch.concat([epochLoss,torch.tensor([loss.item()])],dim=0)
        epochYhat=torch.concat([epochYhat,y_hat.detach().cpu()],dim=0)
    return epochY,epochYhat,epochLoss
    

def evalOneEpoch(net,dataloader):
    net.eval()
    epochY=torch.tensor([])
    epochYhat=torch.tensor([])
    with torch.no_grad():
        for i,data in enumerate(dataloader):
            y,y_hat=forwardBatch(net,data)
            epochY=torch.concat([epochY,y],dim=0)
            epochYhat=torch.concat([epochYhat,y_hat],dim=0)
    return epochY,epochYhat

def trainEpochs(epochNum,net,traindataloader,testdataloader,optimizer,lossFunction):
    for i in range(epochNum):
        trainEpochY,trainEpochYhat,trainEpochLoss=trainOneEpoch(net,traindataloader,optimizer,lossFunction)
        testEpochY,testEpochYhat=evalOneEpoch(net,testdataloader)
    pass