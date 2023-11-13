from . import FolderTree
import os
import mne
import numpy as np
import torch
import pandas as pd
import pywt

from datetime import datetime

from project.dataprocess.SignalProcess import getCWTImage,SequenceFFT,SignalBandPower
##滤波
##小波变换
##delta,beta,theta,gamma波


##小波变换
def getCWTFiles():
    '''
    获取每个文件，每个通道的小波分析图
    输入文件:滤波后的文件（没有归一化）
    输出文件:小波变换结果（归一化的）
    '''
    outputPathDict=FolderTree.getOutPath()
    #print(output_path_dict.keys())
    filted_file_list=os.listdir(outputPathDict['filted_path'])
    for file_name in filted_file_list:
        frames=[]
        name,filetype=file_name.split(".")
        ID,LEVEL,TYPE=name.split("-")
        if TYPE=='EEG':
            EEGfilted=mne.io.read_raw_fif(os.path.join(outputPathDict['filted_path'],file_name),verbose=False)#滤波后的数据
            for channels in EEGfilted.info["ch_names"]:
                ##获取cwt_img
                t,frequencies,log_cwtmatr_uniform=getCWTImage(data=EEGfilted[channels],sfreq=EEGfilted.info["sfreq"])
                frames.append(log_cwtmatr_uniform)
                pass
            data_dict={'time':t,"frequencies":frequencies,"sfreq":EEGfilted.info["sfreq"],"frame_data":torch.stack(frames).to(torch.float32)}
            torch.save(data_dict,os.path.join(outputPathDict['cwt_path'],f"{name}.pth"))


## 滤波后归一化
def Uniform(filtedFolder,uniformFiltedFolder):
    '''
    滤波后归一化
    输入文件是滤波后的文件（没有归一化）
    '''
    filted_path=filtedFolder#滤波后文件夹
    psg_uniformfilted=uniformFiltedFolder#滤波后文件夹
    # conn=sqlite3.connect(os.path.join(psg_uniformfilted,"EEG.db3"))
    filed_filename_list=os.listdir(filted_path)#滤波后文件夹内的文件

    for filted_file_name in filed_filename_list:        
        name,_=filted_file_name.split(".")
        Id,LEVEL,Type=name.split("-")
        if Type=="EEG":
            EEG_filted_file=mne.io.read_raw_fif(os.path.join(filted_path,filted_file_name))
            EEG_channel_dict={}
            ##channel_loop
            for channel_name in EEG_filted_file.info["ch_names"]:
                #归一化
                EEG_channel_dict[channel_name]=((EEG_filted_file[channel_name][0][0]-min(EEG_filted_file[channel_name][0][0]))/(max(EEG_filted_file[channel_name][0][0])-min(EEG_filted_file[channel_name][0][0])))
            info=mne.create_info(ch_names=EEG_filted_file.info["ch_names"],ch_types=["eeg"] * 5,sfreq=EEG_filted_file.info["sfreq"])
            info.set_montage("standard_1020")
            #'Fz', 'Cz', 'C3', 'C4', 'Pz'
            data=np.array([
                EEG_channel_dict['Fz'],EEG_channel_dict['Cz'],EEG_channel_dict['C3'],EEG_channel_dict['C4'],EEG_channel_dict['Pz']
            ])
            raw = mne.io.RawArray(data, info)
            raw.save(os.path.join(psg_uniformfilted,filted_file_name),overwrite=True)

            # pd.DataFrame(EEG_channel_dict).to_sql(f"{name}",conn)



##滤波
def getFilteddFif(psgFolder,outputFolder):
    '''
    获取滤波文件，分别是EEG和ECG文件，EEG滤波范围是0.1-40,ECG是0.1-6
    '''
    PsgPath=psgFolder
    PsgFileList=os.listdir(PsgPath)
    for edf_file_name in PsgFileList:
        
        name,_=edf_file_name.split(".")
        RawEdfFile=mne.io .read_raw_edf(os.path.join(PsgPath,edf_file_name))

        EEGRaw=RawEdfFile.copy().load_data().pick_channels(['Fz', 'Cz', 'C3', 'C4', 'Pz']).filter(0.1,40)
        #EEGRaw=EEGRaw.set_montage(montage=mne.channels.get_builtin_montages()[1],on_missing="ignore")

        ECGRaw=RawEdfFile.copy().load_data().pick_channels(["ECG"]).filter(0.1,6)
        #ECGRaw=ECGRaw.set_montage(montage=mne.channels.get_builtin_montages()[1],on_missing="ignore")

        EEGName=os.path.join(outputFolder,name+"-EEG.fif")
        ECGName=os.path.join(outputFolder,name+"-ECG.fif")

        EEGRaw.save(EEGName,overwrite=True)
    pass


def MakeNonCrossTable(psg_folder,kss_file_path,N_FFT=15*512):
    '''
    获取实验数据：实验人员ID,实验次序，每次反应测试的间隔起始时间、间隔终结时间和反应时长
    输出pandas.Dataframe,columns=["ID","LEVEL","KSS","START", "END","SPEED"]
    '''
    
    date_format = '%Y-%m-%d_%H.%M.%S.%f'
    df=pd.DataFrame(columns=["ID","LEVEL","KSS","START", "END"])
    edfFilenameList=os.listdir(psg_folder)
    for edfFilename in edfFilenameList:
        dict={}
        ID_LEVE,Type=edfFilename.split('.')
        ID,LEVEL=ID_LEVE.split('-')
        try:
            KSS_table=pd.read_csv(kss_file_path,sep=" ",header=None)
            KSS=KSS_table.iloc[int(ID)-1][int(LEVEL)-1]
        except ValueError:
            continue
        edfFile=mne.io.read_raw_edf(os.path.join(psg_folder,edfFilename),verbose=False)      
        
        startIndex=torch.range(0,((len(edfFile)-N_FFT)//N_FFT))*N_FFT
        startTime=startIndex/512
        endIndex=startIndex+N_FFT
        endTime=endIndex/512
        ID=torch.ones_like(startTime)*int(ID)
        LEVEL=torch.ones_like(startTime)*int(LEVEL)
        KSS=torch.ones_like(startTime)*int(KSS)
        dict['ID']=ID
        dict['LEVEL']=LEVEL
        dict["KSS"]=KSS
        dict["START"]=startTime
        dict["END"]=endTime
        tempdf=pd.DataFrame(dict)
        print(tempdf.shape)
        df=pd.concat([df,tempdf],ignore_index=True)
    print(df)
    return df
    pass
def MakePeriodTable(pvt_rt_path,kss_file_path):
    '''
    获取实验数据：实验人员ID,实验次序，每次反应测试的间隔起始时间、间隔终结时间和反应时长
    输出pandas.Dataframe,columns=["ID","LEVEL","KSS","START", "END","SPEED"]
    '''
    pvt_rt_list=os.listdir(pvt_rt_path)
    date_format = '%Y-%m-%d_%H.%M.%S.%f'
    df=pd.DataFrame(columns=["ID","LEVEL","KSS","START", "END","SPEED"])
    for pvt_rt_file_index in range(0,len(pvt_rt_list)):
        file_name=pvt_rt_list[pvt_rt_file_index]
        Id_level=file_name.split("-")
        ##ID
        Id=Id_level[0]
        ##LEVEL
        level=Id_level[1].split(".")[0]
        ##KSS
        try:
            KSS_table=pd.read_csv(kss_file_path,sep=" ",header=None)
            KSS=KSS_table.iloc[int(Id)-1][int(level)-1]
        except ValueError:
            continue

        pvr_rt_file=pvr_rt_file=pd.read_csv(os.path.join(pvt_rt_path,file_name))

        zero_time=pvr_rt_file.columns.to_list()[0]
        zero_time_datetime_obj=datetime.strptime(zero_time, date_format)
        zero_timestamp=zero_time_datetime_obj.timestamp()

        for pvrt_rt_file_index in range(len(pvr_rt_file)-1):
            record=pvr_rt_file.iloc[pvrt_rt_file_index,0]
            time_split=record.split(";")
            start_time,end_time=time_split[0],time_split[1]
            start_datetime_obj = datetime.strptime(start_time, date_format)
            end_datetime_obj = datetime.strptime(end_time, date_format)
            
            start_timestamp = start_datetime_obj.timestamp()
            end_timestamp = end_datetime_obj.timestamp()

            
            if pvrt_rt_file_index==0:
                temp_start=0
                temp_end=start_timestamp-zero_timestamp
                buffer=end_timestamp-zero_timestamp
            else:
                temp_start=buffer
                temp_end=start_timestamp-zero_timestamp
                buffer=end_timestamp-zero_timestamp
            speed=end_timestamp-start_timestamp
            df.loc[len(df)]={"ID":Id,"LEVEL":level,"KSS":KSS,"START":temp_start,"END":temp_end,"SPEED":speed}
            pass
        pass
    return df

##
def getBandPower(infoTableFilePaht,kssFilePath,psdFolderPath,bandaPowerFilePath,N_FFT=15*512,HOP_LEN=5*512):
    '''
    OutputFolderDict['info_file']
    DatasetFolderDict['kss_file']
    DatasetFolderDict['psg_folder']
    OutputFolderDict['bandpower_file']
    '''
    # infoTableFilePaht=
    # kssFilePath=
    # psdFolderPath=
    # bandaPowerFilePath=

    WaveDict={
    'delta':{'LOW':1,"HIGH":3},
    'theta':{'LOW':4,"HIGH":7},
    'alpha':{'LOW':8,"HIGH":11},
    'beta':{'LOW':12,"HIGH":29},
    'gamma':{'LOW':30,"HIGH":100},
    }
    infoTable=pd.read_csv(infoTableFilePaht)
    kssTable=pd.read_csv(kssFilePath,sep=' ',header=None)
    for filename in os.listdir(psdFolderPath):
    #读数据
        FiltedEdf=mne.io.read_raw_edf(os.path.join(psdFolderPath,filename))
        FiltedEdf=FiltedEdf.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz'])
        #提取文件名中的信息
        filename,_=filename.split(".")
        ID,LEVEL=filename.split('-')
        ID=int(ID)
        LEVEL=int(LEVEL)
        #取KSS评分
        KSS=int(kssTable.iloc[int(ID)-1][int(LEVEL)-1])

        ##channel_loop
        PowerDict={}
        for ch_name in FiltedEdf.info['ch_names']:
            y=FiltedEdf[ch_name][0][0].T
            sfft,freq=SequenceFFT(y,N_FFT,HOP_LEN)
            ##band_loop
            temp=[] 
            for bandName,band in list(WaveDict.items()):
                # print(bandName,band)
                idx_band=np.logical_and(WaveDict[bandName]['LOW']<=freq,freq<=WaveDict[bandName]['HIGH'])
                if bandName=='gamma':
                    idx_band=np.logical_and(idx_band, freq != 50)
                    pass
                bandPower=np.sum(abs(sfft[idx_band,:])**2*2,axis=0)
                # print(bandPower.shape)
                PowerDict[ch_name+'_'+bandName]=bandPower
                temp.append(bandPower)
                
                if  bandName==list(WaveDict.keys())[-1]:
                    allPower=np.sum(temp,axis=0)
                    # print(allPower.shape)               
        PowerDict["ID"]=np.ones_like(bandPower)*ID  
        PowerDict["LEVEL"]=np.ones_like(bandPower)*LEVEL
        PowerDict["KSS"]=np.ones_like(bandPower)*KSS     
        
        rawPowerTable=pd.DataFrame(PowerDict)
        rawPowerTable.to_csv(bandaPowerFilePath, mode='a', header=not os.path.exists(bandaPowerFilePath), index=False)

def getUniformBandPower(bandPowerTablePath,uniformBandPowerTablePath):
    
    rawPowerTable=pd.read_csv(bandPowerTablePath)
    ch_names=['Fz', 'Cz', 'C3', 'C4', 'Pz']

    table=[]
    for ch_name in  ch_names:
        table.append(rawPowerTable.filter(like=ch_name).div(rawPowerTable.filter(like=ch_name).sum(axis=1),axis=0))
        merged_df = pd.concat(table, axis=1)
        merged_df = pd.concat((merged_df,rawPowerTable[["ID","LEVEL","KSS"]]),axis=1)
        merged_df.to_csv(uniformBandPowerTablePath)
    return merged_df
    pass     
        

    
def getSFFT(psgFolder,outputFolder,N_FFT=15*512,HOP_LEN=5*512):
    '''
    
    '''
    PsgPath=psgFolder
    PsgFileList=os.listdir(PsgPath)
    for edf_file_name in PsgFileList:   
        name,_=edf_file_name.split(".")
        RawEdfFile=mne.io .read_raw_edf(os.path.join(PsgPath,edf_file_name))
        EEGRaw=RawEdfFile.copy().load_data().pick(['Fz', 'Cz', 'C3', 'C4', 'Pz'])
        
        for ch_name in EEGRaw.info['ch_names']:
            y=EEGRaw[ch_name][0][0].T
            sfft,freq=SequenceFFT(y,N_FFT,HOP_LEN)
            
    pass