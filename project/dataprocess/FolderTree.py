import os 
data_root="/mount/mount_dataset/driver_dataset/DROZY/DROZY"
output_root="/mount/mount_project/output_data/"
def getOutPath(output_root="/mount/mount_project/output_data/"):
    '''
    输入：输出文件夹的根目录
    输出：输出文件夹下文件和文件夹的路径
        info_filename：["ID","LEVEL","KSS","START", "END","SPEED"] csv文件
        filted_path：滤波后的序列文件
        nomalfiled_path:滤波后经过标准化过程的文件y=E((X-E(X))^2)
        uniformfiled_path:滤波后经过归一化后的文件y=(x-min(x))/(max(x)-min(x))
        cwt:经过小波变换后的数据文件
    '''

    
    filted_path="psg_filted"
    nomalfilted_path="psg_nomalfilted"
    uniformfilted_path="psg_uniformfilted"
    cwt_path="cwt"
    info_filename="info.csv"
    bandPowerFilename='bandpower.csv'
    uniformBandPowerFilename='uniformbandpower.csv'
    EEG_table="EEG_table.csv"
    ECG_table="ECG_table.csv"

    noncross='noncross.csv'
    noncross_file_path=os.path.join(data_root,noncross)

    output_path_dict={
        "info_file":os.path.join(output_root,info_filename),
        "filted_folder":os.path.join(output_root,filted_path),
        "nomalfilted_folder":os.path.join(output_root,nomalfilted_path),
        "uniformfilted_folder":os.path.join(output_root,uniformfilted_path),
        "cwt_folder":os.path.join(output_root,cwt_path),

        "EEG_table_file":os.path.join(output_root,EEG_table),
        "ECG_table_file":os.path.join(output_root,ECG_table),

        'bandpower_file':os.path.join(output_root,bandPowerFilename),
        'uniformbandpower_file':os.path.join(output_root,uniformBandPowerFilename),
        'noncross_file':noncross_file_path
    }    
    return output_path_dict

def getDataPath(data_root="/mount/mount_project/output_data/"):
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
        "psg_folder":psg_path,
        "pvt_rt_folder":pvt_rt_path,
        "videos_folder":videos_path,
        "annotation_auto_folder":annotation_auto_path,
        "annotation_manual_folder":annotation_manual_path,
        "kss_file":kss_file_path,
        "kinect_file":kinect_file_path,
        
    }
    file_path_array=[psg_path,pvt_rt_path,videos_path,annotation_auto_path,annotation_manual_path,kss_file_path,kinect_file_path]
    return file_path_dict
    pass
def checkFolderTree(dataRoot,outRoot):
    datasetDict=getDataPath(dataRoot)
    outPathDict=getOutPath(outRoot)
    print("Check Dataset Path")
    for key in datasetDict.keys():
        print(datasetDict[key],os.path.exists(datasetDict[key]))
    print("Check Output Path")
    for key in outPathDict.keys():
        
        if ( key.split('_')[-1]=='folder') and (not os.path.exists(outPathDict[key])):
            os.makedirs(outPathDict[key])
        print(outPathDict[key],os.path.exists(outPathDict[key]))