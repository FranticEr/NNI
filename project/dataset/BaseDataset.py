from torch.utils.data import Dataset
import torch
from typing import Any

import torch
from torch.utils.data import*
import os
import pandas as pd
import mne

class TableDataset(Dataset):
    def __init__(self,table:pd.DataFrame,label:str,drop:list) -> None:
        super().__init__()
        self.Table=table
        self.DropColums=drop
        self.LableColums=label
        self.InputTable=self.Table.drop(columns=self.DropColums+[self.LableColums])
        self.LableTable=self.Table[self.LableColums]
    def __len__(self):
        return len(self.Table)
    def __getitem__(self, index) -> Any:
        x=torch.tensor( self.InputTable.iloc[index].values).to(torch.float32)
        y=torch.tensor(self.LableTable.iloc[index]).long()
        return x,y
    
class TableControlFifDataset(Dataset):
    def __init__(self,info_table,psgFileFolderPath,Mintime=5,fix_len=True) -> None:
        super().__init__()
        '''
        与table_control_dataset类不同点是，此类会在读取数据时将数据加载到内存中，存储在temp_file_dict字典中
        '''
        ##在这可以控制训练集和测试集
        self.InfoTable=info_table
        self.PsgFileFolderPath=psgFileFolderPath

        self.Mintime=Mintime
        self.fix_len=fix_len
        self.temp_file_name=None
        self.TempFileDict={}
        
        self.InfoTable=self.InfoTable[self.InfoTable['START']>Mintime]
        self.len=len(self.InfoTable)

    def __len__(self):
        return self.len
        pass
    def GetPsgFile(self,fullfilename,start,end,fix_len=True):
        if not fullfilename in self.TempFileDict:
            self.PsgFile = mne.io.read_raw_fif(fullfilename,verbose=False)
            self.TempFileDict[fullfilename]=self.PsgFile
        else :
            self.PsgFile=self.TempFileDict[fullfilename]
        sfreq=self.PsgFile.info['sfreq']
        self.PsgFile=self.PsgFile.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz'])
        if fix_len==True:
            data,time=self.PsgFile[:,int(sfreq*(end-self.Mintime)):int(sfreq*end)]
            pass
        else:
            data,time=self.PsgFile[:,int(sfreq*start):int(sfreq*end)]
            pass       
        return data,time
        pass
    def __getitem__(self, index):
        info=self.InfoTable.iloc[index]
        ID,LEVEL,KSS,START,END=info["ID"],info["LEVEL"],info["KSS"],info["START"],info["END"]
        ##拼接文件名
        filename=str(int(ID))+"-"+str(int(LEVEL))+"-EEG.fif"
        fullfilename=os.path.join(self.PsgFileFolderPath, filename)
        #psg_file_name=os.path.join(self.out_path_dict["uniformfilted_path"],file_name)
        data,time=self.GetPsgFile(fullfilename,START,END,fix_len=True)
        index=['Fz', 'Cz', 'C3', 'C4', 'Pz']
        return {"data":data,"time":time,"index":index,"KSS":KSS,"LEVEL":LEVEL}