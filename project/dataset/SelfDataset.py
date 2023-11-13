import torch
from torch.utils.data import*
import os
import pandas as pd
import mne
class edf_map_dataset(Dataset):
    def __init__(self,data_root,output_root,Mintime=5,fix_len=True) -> None:
        super().__init__()
        ##文件目录
        self.data_path_dict=self.get_data_path(data_root=data_root)
        self.out_path_dict=self.get_out_path(output_root=output_root)
        ##在这可以控制训练集和测试集
        self.info_table=pd.read_csv(self.out_path_dict["info_file"])
        self.Mintime=Mintime
        self.fix_len=fix_len
        self.temp_file_name=None
        
        self.info_table=self.info_table[self.info_table['START']>Mintime]
        self.len=len(self.info_table)

    def __len__(self):
        return self.len
        pass
    def get_psg(self,file_name,start,end,fix_len=True):
        if self.temp_file_name==file_name or self.temp_file_name==None:
            self.temp_file_name=file_name
            self.psg_file = mne.io.read_raw_fif(os.path.join(self.out_path_dict["uniformfilted_path"], file_name),verbose=False)
            
          
        sfreq=self.psg_file.info['sfreq']
        self.psg_file=self.psg_file.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz'])
        if fix_len==True:
            data,time=self.psg_file[:,int(sfreq*(end-self.Mintime)):int(sfreq*end)]
            pass
        else:
            data,time=self.psg_file[:,int(sfreq*start):int(sfreq*end)]
            pass       
        return data,time
        pass
    def get_videos(file_name):
        pass

    def __getitem__(self, index):
        info=self.info_table.iloc[index]
        ID,LEVEL,KSS,START,END,SPEED=info["ID"],info["LEVEL"],info["KSS"],info["START"],info["END"],info["SPEED"]
        ##拼接文件名
        file_name=str(int(ID))+"-"+str(int(LEVEL))+"-EEG.fif"
        #psg_file_name=os.path.join(self.out_path_dict["uniformfilted_path"],file_name)
        data,time=self.get_psg(file_name,START,END,self.fix_len)
        index=a=['Fz', 'Cz', 'C3', 'C4', 'Pz']
        return {"data":data,"time":time,"index":index,"KSS":KSS,"LEVEL":LEVEL,"SPEED":SPEED}
    

    def get_data_path(self,data_root="/mount/mount_dataset/driver_dataset/DROZY/DROZY"):
        
        # return file_path_dict
        pass
    def get_out_path(self,output_root="/mount/mount_project/output_data/"):
        '''
        输入：输出文件夹的根目录
        输出：输出文件夹下文件和文件夹的路径
            info_filename：["ID","LEVEL","KSS","START", "END","SPEED"] csv文件
            filted_path：滤波后的序列文件
            nomalfiled_path:滤波后经过标准化过程的文件y=E((X-E(X))^2)
            uniformfiled_path:滤波后经过归一化后的文件y=(x-min(x))/(max(x)-min(x))
            cwt:经过小波变换后的数据文件
        '''

       
        # return output_path_dict
        pass

import torch
from torch.utils.data import*
import os
import pandas as pd
import mne
class table_control_dataset(Dataset):
    def __init__(self,info_table,data_root,output_root,Mintime=5,fix_len=True) -> None:
        super().__init__()
        ##文件目录
        self.data_path_dict=self.get_data_path(data_root=data_root)
        self.out_path_dict=self.get_out_path(output_root=output_root)
        ##在这可以控制训练集和测试集
        self.info_table=info_table
        self.Mintime=Mintime
        self.fix_len=fix_len
        self.temp_file_name=None
        
        self.info_table=self.info_table[self.info_table['START']>Mintime]
        self.len=len(self.info_table)

    def __len__(self):
        return self.len
        pass
    def get_psg(self,file_name,start,end,fix_len=True):
        if self.temp_file_name==file_name or self.temp_file_name==None:
            self.temp_file_name=file_name
            self.psg_file = mne.io.read_raw_fif(os.path.join(self.out_path_dict["uniformfilted_path"], file_name),verbose=False)
            
          
        sfreq=self.psg_file.info['sfreq']
        self.psg_file=self.psg_file.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz'])
        if fix_len==True:
            data,time=self.psg_file[:,int(sfreq*(end-self.Mintime)):int(sfreq*end)]
            pass
        else:
            data,time=self.psg_file[:,int(sfreq*start):int(sfreq*end)]
            pass       
        return data,time
        pass
    def get_videos(file_name):
        pass

    def __getitem__(self, index):
        info=self.info_table.iloc[index]
        ID,LEVEL,KSS,START,END,SPEED=info["ID"],info["LEVEL"],info["KSS"],info["START"],info["END"],info["SPEED"]
        ##拼接文件名
        file_name=str(int(ID))+"-"+str(int(LEVEL))+"-EEG.fif"
        #psg_file_name=os.path.join(self.out_path_dict["uniformfilted_path"],file_name)
        data,time=self.get_psg(file_name,START,END,self.fix_len)
        index=a=['Fz', 'Cz', 'C3', 'C4', 'Pz']
        return {"data":data,"time":time,"index":index,"KSS":KSS,"LEVEL":LEVEL,"SPEED":SPEED}
    

    def get_data_path(self,data_root="/mount/mount_dataset/driver_dataset/DROZY/DROZY"):
        '''data_root:drozy数据集的顶级目录，psg,videos_i8的上级目录
        返回各个子目录的路径,dict，词典格式
        '''
        annotation_auto="annotations-auto"
        annotation_manual="annotations-manual"
        psg="psg"
        pvt_rt="pvt-rt"
        videos="videos_i8"
        kinect_intrinsics="kinect-intrinsics.yaml"
        kss="KSS.txt"

        psg_path=os.path.join(data_root,psg)
        pvt_rt_path=os.path.join(data_root,pvt_rt)
        videos_path=os.path.join(data_root,videos)
        annotation_auto_path=os.path.join(data_root,annotation_auto)
        annotation_manual_path=os.path.join(data_root,annotation_manual)
        kinect_file_path=os.path.join(data_root,kinect_intrinsics)
        kss_file_path=os.path.join(data_root,kss)

        file_path_dict={
            "psg_path":psg_path,
            "pvt_rt_path":pvt_rt_path,
            "videos_path":videos_path,
            "annotation_auto_path":annotation_auto_path,
            "annotation_manual_path":annotation_manual_path,
            "kss_file_path":kss_file_path,
            "kinect_file_path":kinect_file_path
        }
        file_path_array=[psg_path,pvt_rt_path,videos_path,annotation_auto_path,annotation_manual_path,kss_file_path,kinect_file_path]
        return file_path_dict
    def get_out_path(self,output_root="/mount/mount_project/output_data/"):
        '''
        输入：输出文件夹的根目录
        输出：输出文件夹下文件和文件夹的路径
            info_filename：["ID","LEVEL","KSS","START", "END","SPEED"] csv文件
            filted_path：滤波后的序列文件
            nomalfiled_path:滤波后经过标准化过程的文件y=E((X-E(X))^2)
            uniformfiled_path:滤波后经过归一化后的文件y=(x-min(x))/(max(x)-min(x))
            cwt:经过小波变换后的数据文件
        '''

        info_filename="info.csv"
        filted_path="psg_filted"
        nomalfilted_path="psg_nomalfilted"
        uniformfilted_path="psg_uniformfilted"
        cwt_path="cwt"

        output_path_dict={
            "info_file":os.path.join(output_root,info_filename),
            "filted_path":os.path.join(output_root,filted_path),
            "nomalfilted_path":os.path.join(output_root,nomalfilted_path),
            "uniformfilted_path":os.path.join(output_root,uniformfilted_path),
            "cwt_path":os.path.join(output_root,cwt_path)
        }
        #print(output_path_dict)
        return output_path_dict
        pass

from project.dataprocess import FolderTree

class TableControlFullLoadDataset(Dataset):
    '''
    获取滤波信号
    infoTable(outputFolder)->getEDFFile(DatasetFolder)->getsignal[start,stop]
    '''
    def __init__(self,info_table,data_root,output_root,Mintime=5,fix_len=True) -> None:
        super().__init__()
        '''
        与table_control_dataset类不同点是，此类会在读取数据时将数据加载到内存中，存储在temp_file_dict字典中
        '''
        ##文件目录
        self.data_path_dict=self.getDataPath(data_root=data_root)
        self.out_path_dict=self.getOutPath(output_root=output_root) 
        ##在这可以控制训练集和测试集

        self.info_table=info_table
        self.Mintime=Mintime
        self.fix_len=fix_len
        self.temp_file_name=None
        self.temp_file_dict={}

        self.info_table=self.info_table[self.info_table['START']>Mintime]
        self.len=len(self.info_table)

    def __len__(self):
        return self.len
        pass
    def GetPsgFile(self,file_name,start,end,fix_len=True):
        if not file_name in self.temp_file_dict:
            self.psg_file = mne.io.read_raw_fif(os.path.join(self.out_path_dict["uniformfilted_folder"], file_name),verbose=False)
            self.temp_file_dict[file_name]=self.psg_file
        else :
            self.psg_file=self.temp_file_dict[file_name]
        sfreq=self.psg_file.info['sfreq']
        self.psg_file=self.psg_file.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz'])
        if fix_len==True:
            data,time=self.psg_file[:,int(sfreq*(end-self.Mintime)):int(sfreq*end)]
            pass
        else:
            data,time=self.psg_file[:,int(sfreq*start):int(sfreq*end)]
            pass       
        return data,time
        pass
    def get_videos(file_name):
        pass

    def __getitem__(self, index):
        info=self.info_table.iloc[index]
        ID,LEVEL,KSS,START,END,SPEED=info["ID"],info["LEVEL"],info["KSS"],info["START"],info["END"],info["SPEED"]
        ##拼接文件名
        file_name=str(int(ID))+"-"+str(int(LEVEL))+"-EEG.fif"
        #psg_file_name=os.path.join(self.out_path_dict["uniformfilted_path"],file_name)
        data,time=self.GetPsgFile(file_name,START,END,self.fix_len)
        idx=['Fz', 'Cz', 'C3', 'C4', 'Pz']
        return {"data":data,"time":time,"index":idx,"KSS":KSS,"LEVEL":LEVEL,"SPEED":SPEED} 
    def getDataPath(self,data_root="/mount/mount_dataset/driver_dataset/DROZY/DROZY"):
        '''data_root:drozy数据集的顶级目录，psg,videos_i8的上级目录
        返回各个子目录的路径,dict，词典格式
        '''
        file_path_dict=FolderTree.getDataPath(data_root)
        return file_path_dict
    def getOutPath(self,output_root="/mount/mount_project/output_data/"):
        '''
        输入：输出文件夹的根目录
        输出：输出文件夹下文件和文件夹的路径
            info_filename：["ID","LEVEL","KSS","START", "END","SPEED"] csv文件
            filted_path：滤波后的序列文件
            nomalfiled_path:滤波后经过标准化过程的文件y=E((X-E(X))^2)
            uniformfiled_path:滤波后经过归一化后的文件y=(x-min(x))/(max(x)-min(x))
            cwt:经过小波变换后的数据文件
        '''
        output_path_dict=FolderTree.getOutPath(output_root)
        return output_path_dict
        pass
class NonCrossDataset(Dataset):
    def __init__(self,info_table,data_root,output_root,Mintime=5,fix_len=True) -> None:
        super().__init__()
        '''
        与table_control_dataset类不同点是，此类会在读取数据时将数据加载到内存中，存储在temp_file_dict字典中
        '''
        ##文件目录
        self.data_path_dict=self.getDataPath(data_root=data_root)
        self.out_path_dict=self.getOutPath(output_root=output_root)

        
        ##在这可以控制训练集和测试集
        self.info_table=info_table
        self.Mintime=Mintime
        self.fix_len=fix_len
        self.temp_file_name=None
        self.temp_file_dict={}
        self.info_table=self.info_table[self.info_table['START']>Mintime]
        self.len=len(self.info_table)

    def __len__(self):
        return self.len
        pass
    def GetPsgFile(self,file_name,start,end,fix_len=True):
        if not file_name in self.temp_file_dict:
            self.psg_file = mne.io.read_raw_fif(os.path.join(self.out_path_dict["uniformfilted_folder"], file_name),verbose=False)
            self.temp_file_dict[file_name]=self.psg_file
        else :
            self.psg_file=self.temp_file_dict[file_name]
        sfreq=self.psg_file.info['sfreq']
        self.psg_file=self.psg_file.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz'])
        if fix_len==True:
            data,time=self.psg_file[:,int(sfreq*(end-self.Mintime)):int(sfreq*end)]
            pass
        else:
            data,time=self.psg_file[:,int(sfreq*start):int(sfreq*end)]
            pass       
        return data,time
        pass
    def get_videos(file_name):
        pass

    def __getitem__(self, index):
        info=self.info_table.iloc[index]
        ID,LEVEL,KSS,START,END,SPEED=info["ID"],info["LEVEL"],info["KSS"],info["START"],info["END"]
        ##拼接文件名
        file_name=str(int(ID))+"-"+str(int(LEVEL))+"-EEG.fif"
        #psg_file_name=os.path.join(self.out_path_dict["uniformfilted_path"],file_name)
        data,time=self.GetPsgFile(file_name,START,END,fix_len=True)
        index=['Fz', 'Cz', 'C3', 'C4', 'Pz']
        return {"data":data,"time":time,"index":index,"KSS":KSS,"LEVEL":LEVEL}
    

    
    def getDataPath(self,data_root="/mount/mount_dataset/driver_dataset/DROZY/DROZY"):
        '''data_root:drozy数据集的顶级目录，psg,videos_i8的上级目录
        返回各个子目录的路径,dict，词典格式
        '''
        file_path_dict=FolderTree.getDataPath(data_root)
        return file_path_dict
    def getOutPath(self,output_root="/mount/mount_project/output_data/"):
        '''
        输入：输出文件夹的根目录
        输出：输出文件夹下文件和文件夹的路径
            info_filename：["ID","LEVEL","KSS","START", "END","SPEED"] csv文件
            filted_path：滤波后的序列文件
            nomalfiled_path:滤波后经过标准化过程的文件y=E((X-E(X))^2)
            uniformfiled_path:滤波后经过归一化后的文件y=(x-min(x))/(max(x)-min(x))
            cwt:经过小波变换后的数据文件
        '''
        output_path_dict=FolderTree.getOutPath(output_root)
        return output_path_dict
        pass

class TableControlFullLoadDatasetIndex(Dataset):
    def __init__(self,info_table,data_root,output_root,Mintime=5,fix_len=True) -> None:
        super().__init__()
        '''
        与table_control_dataset类不同点是，此类会在读取数据时将数据加载到内存中，存储在temp_file_dict字典中
        '''
        ##文件目录
        self.data_path_dict=self.getDataPath(data_root=data_root)
        self.out_path_dict=self.getOutPath(output_root=output_root)

        
        ##在这可以控制训练集和测试集
        self.info_table=info_table
        self.Mintime=Mintime
        self.fix_len=fix_len
        self.temp_file_name=None
        self.temp_file_dict={}
        self.info_table=self.info_table[self.info_table['START']>Mintime]
        self.len=len(self.info_table)

    def __len__(self):
        return self.len
        pass
    def GetPsgFile(self,file_name,start,end,fix_len=True):
        if not file_name in self.temp_file_dict:
            self.psg_file = mne.io.read_raw_fif(os.path.join(self.out_path_dict["uniformfilted_folder"], file_name),verbose=False)
            self.temp_file_dict[file_name]=self.psg_file
        else :
            self.psg_file=self.temp_file_dict[file_name]
        sfreq=self.psg_file.info['sfreq']
        self.psg_file=self.psg_file.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz'])
        if fix_len==True:
            data,time=self.psg_file[:,int(sfreq*(end-self.Mintime)):int(sfreq*end)]
            pass
        else:
            data,time=self.psg_file[:,int(sfreq*start):int(sfreq*end)]
            pass       
        return data,time
        pass
    def get_videos(file_name):
        pass

    def __getitem__(self, index):
        info=self.info_table.iloc[index]
        ID,LEVEL,KSS,START,END,SPEED=info["ID"],info["LEVEL"],info["KSS"],info["START"],info["END"],info["SPEED"]
        ##拼接文件名
        file_name=str(int(ID))+"-"+str(int(LEVEL))+"-EEG.fif"
        #psg_file_name=os.path.join(self.out_path_dict["uniformfilted_path"],file_name)
        data,time=self.GetPsgFile(file_name,START,END,self.fix_len)
        idx=['Fz', 'Cz', 'C3', 'C4', 'Pz']
        return {"data":data,"time":time,"index":idx,"KSS":KSS,"LEVEL":LEVEL,"SPEED":SPEED} ,index
    def getDataPath(self,data_root="/mount/mount_dataset/driver_dataset/DROZY/DROZY"):
        '''data_root:drozy数据集的顶级目录，psg,videos_i8的上级目录
        返回各个子目录的路径,dict，词典格式
        '''
        file_path_dict=FolderTree.getDataPath(data_root)
        return file_path_dict
    def getOutPath(self,output_root="/mount/mount_project/output_data/"):
        '''
        输入：输出文件夹的根目录
        输出：输出文件夹下文件和文件夹的路径
            info_filename：["ID","LEVEL","KSS","START", "END","SPEED"] csv文件
            filted_path：滤波后的序列文件
            nomalfiled_path:滤波后经过标准化过程的文件y=E((X-E(X))^2)
            uniformfiled_path:滤波后经过归一化后的文件y=(x-min(x))/(max(x)-min(x))
            cwt:经过小波变换后的数据文件
        '''
        output_path_dict=FolderTree.getOutPath(output_root)
        return output_path_dict
        pass
class table_control_dataset_fullload(Dataset):
    def __init__(self,info_table,data_root,output_root,Mintime=5,fix_len=True) -> None:
        super().__init__()
        '''
        与table_control_dataset类不同点是，此类会在读取数据时将数据加载到内存中，存储在temp_file_dict字典中
        '''
        ##文件目录
        self.data_path_dict=self.get_data_path(data_root=data_root)
        self.out_path_dict=self.get_out_path(output_root=output_root)

        
        ##在这可以控制训练集和测试集
        self.info_table=info_table
        self.Mintime=Mintime
        self.fix_len=fix_len
        self.temp_file_name=None
        self.temp_file_dict={}
        self.info_table=self.info_table[self.info_table['START']>Mintime]
        self.len=len(self.info_table)

    def __len__(self):
        return self.len
        pass
    def GetPsgFile(self,file_name,start,end,fix_len=True):
        if not file_name in self.temp_file_dict:
            self.psg_file = mne.io.read_raw_fif(os.path.join(self.out_path_dict["uniformfilted_path"], file_name),verbose=False)
            self.temp_file_dict[file_name]=self.psg_file
        else :
            self.psg_file=self.temp_file_dict[file_name]
        sfreq=self.psg_file.info['sfreq']
        self.psg_file=self.psg_file.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz'])
        if fix_len==True:
            data,time=self.psg_file[:,int(sfreq*(end-self.Mintime)):int(sfreq*end)]
            pass
        else:
            data,time=self.psg_file[:,int(sfreq*start):int(sfreq*end)]
            pass       
        return data,time
        pass
    def get_videos(file_name):
        pass

    def __getitem__(self, index):
        info=self.info_table.iloc[index]
        ID,LEVEL,KSS,START,END,SPEED=info["ID"],info["LEVEL"],info["KSS"],info["START"],info["END"],info["SPEED"]
        ##拼接文件名
        file_name=str(int(ID))+"-"+str(int(LEVEL))+"-EEG.fif"
        #psg_file_name=os.path.join(self.out_path_dict["uniformfilted_path"],file_name)
        data,time=self.GetPsgFile(file_name,START,END,self.fix_len)
        index=a=['Fz', 'Cz', 'C3', 'C4', 'Pz']
        return {"data":data,"time":time,"index":index,"KSS":KSS,"LEVEL":LEVEL,"SPEED":SPEED}
    

    def get_data_path(self,data_root="/mount/mount_dataset/driver_dataset/DROZY/DROZY"):
        '''data_root:drozy数据集的顶级目录，psg,videos_i8的上级目录
        返回各个子目录的路径,dict，词典格式
        '''
        annotation_auto="annotations-auto"
        annotation_manual="annotations-manual"
        psg="psg"
        pvt_rt="pvt-rt"
        videos="videos_i8"
        kinect_intrinsics="kinect-intrinsics.yaml"
        kss="KSS.txt"

        psg_path=os.path.join(data_root,psg)
        pvt_rt_path=os.path.join(data_root,pvt_rt)
        videos_path=os.path.join(data_root,videos)
        annotation_auto_path=os.path.join(data_root,annotation_auto)
        annotation_manual_path=os.path.join(data_root,annotation_manual)
        kinect_file_path=os.path.join(data_root,kinect_intrinsics)
        kss_file_path=os.path.join(data_root,kss)

        file_path_dict={
            "psg_path":psg_path,
            "pvt_rt_path":pvt_rt_path,
            "videos_path":videos_path,
            "annotation_auto_path":annotation_auto_path,
            "annotation_manual_path":annotation_manual_path,
            "kss_file_path":kss_file_path,
            "kinect_file_path":kinect_file_path
        }
        file_path_array=[psg_path,pvt_rt_path,videos_path,annotation_auto_path,annotation_manual_path,kss_file_path,kinect_file_path]
        return file_path_dict
    def get_out_path(self,output_root="/mount/mount_project/output_data/"):
        '''
        输入：输出文件夹的根目录
        输出：输出文件夹下文件和文件夹的路径
            info_filename：["ID","LEVEL","KSS","START", "END","SPEED"] csv文件
            filted_path：滤波后的序列文件
            nomalfiled_path:滤波后经过标准化过程的文件y=E((X-E(X))^2)
            uniformfiled_path:滤波后经过归一化后的文件y=(x-min(x))/(max(x)-min(x))
            cwt:经过小波变换后的数据文件
        '''

        info_filename="info.csv"
        filted_path="psg_filted"
        nomalfilted_path="psg_nomalfilted"
        uniformfilted_path="psg_uniformfilted"
        cwt_path="cwt"

        output_path_dict={
            "info_file":os.path.join(output_root,info_filename),
            "filted_path":os.path.join(output_root,filted_path),
            "nomalfilted_path":os.path.join(output_root,nomalfilted_path),
            "uniformfilted_path":os.path.join(output_root,uniformfilted_path),
            "cwt_path":os.path.join(output_root,cwt_path)
        }
        #print(output_path_dict)
        return output_path_dict
        pass


import pandas as pd
from typing import Any
class DictTableDataset(Dataset):
    def __init__(self,table:pd.DataFrame) -> None:
        super().__init__()
        self.Table=table
    def __len__(self):
        return len(self.Table)
    def __getitem__(self, index) -> Any:
        return self.Table.iloc[index].to_dict()

class TensorTableDataset(Dataset):
    def __init__(self,table:pd.DataFrame) -> None:
        super().__init__()
        self.Table=table
    def __len__(self):
        return len(self.Table)
    def __getitem__(self, index) -> Any:
        # torch.tensor(trainDataTable.drop(columns=['ID','KSS','LEVEL']).iloc[1].values)
        x=torch.tensor(self.Table.drop(columns=['ID','KSS','LEVEL']).iloc[index].values).to(torch.float32)
        y=torch.tensor(self.Table['LEVEL'].iloc[index]).long()-1
        return x,y
#路径

