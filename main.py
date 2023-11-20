import pandas as pd
from project.dataprocess.DatasetProcess import *
kss_table=r'D:\dataset\driver_dataset\DROZY\DROZY\KSS.txt'
kss_table=pd.read_csv(kss_table,sep=" ",header=None)

getKSSFramesFolder(r"D:\dataset\driver_dataset\DROZY\DROZY\videos_i8",r'D:\project_meta\NNproject\NNI\output\video_frames\kss_frames',kss_table)
