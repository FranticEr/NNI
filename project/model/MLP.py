import torch.nn as nn
def MLP(numFeature,interWidth,classes):
    interWidth.insert(0,numFeature)
    interWidth.append(classes)
    net=nn.Sequential()
    for i in range(len(interWidth)-2) :
        net.append(nn.Linear(interWidth[i],interWidth[i+1]))
        #net.append(nn.Dropout(0.5))
        net.append(nn.ReLU())
    net.append(nn.Linear(interWidth[-2],interWidth[-1]))
    return net