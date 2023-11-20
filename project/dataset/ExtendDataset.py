from .BaseDataset import *
import numpy as np
from biosppy.signals import ecg
from biosppy.storage import load_txt
import matplotlib.pyplot as plt
from project.dataprocess.SignalProcess import SequenceFFT,getCWTImage

### EEG EndToEnd Dataset
class LEVELEEGEndToEndDataset(TableControlEEGDataset):
    def __getitem__(self, index):
        self.DataDict= super().__getitem__(index)
        return self.DataDict['data'],self.DataDict['LEVEL']
class KSSEEGEndToEndDataset(TableControlEEGDataset):
    def __getitem__(self, index):
        self.DataDict= super().__getitem__(index)
        return self.DataDict['data'],self.DataDict['KSS']
##EEG Feature Dataset
class SFFTDataset(TableControlEEGDataset):
    def __getitem__(self, index):
        self.DataDict= super().__getitem__(index)
        self.SFFTlist=[]
        for channel in self.DataDict['data']:
            sfft,freq=SequenceFFT(channel,sfreq=512,N_FFT=512,HOP_LENGTH=128)
            abssfft=abs(sfft[256:256+40,:])
            self.SFFTlist.append(abssfft)
        return {'SFFT':self.SFFTlist,'LEVEL':self.DataDict['LEVEL'],'KSS':self.DataDict['KSS']}

class LEVELSFFTDataset(SFFTDataset):
    def __getitem__(self, index):
        self.DataDict= super().__getitem__(index)
        return self.DataDict['SFFT'],self.DataDict['LEVEL']
    
class KSSSFFTDataset(SFFTDataset):
    def __getitem__(self, index):
        self.DataDict= super().__getitem__(index)
        return self.DataDict['SFFT'],self.DataDict['KSS']

### ECG EndToEnd Dataset
class LEVELECGEndToEndDataset(TableControlECGDataset):
    def __getitem__(self, index):
        sfreq=512
        Datadict = super().__getitem__(index)
        signalwave=Datadict['data']
        rpeaks0=ecg.christov_segmenter(signalwave,sfreq)
        rpeaks1=ecg.christov_segmenter(-1*signalwave,sfreq)
        if sum(signalwave[rpeaks0[0]])<sum(-1*signalwave[rpeaks1[0]]):
            signalwave=-1*signalwave
        result=ecg.ecg(signalwave,sfreq)
        result_dict=result.as_dict()
        return result_dict['filtered'],Datadict['LEVEL']
class KSSECGEndToEndDataset(TableControlECGDataset):
    def __getitem__(self, index):
        sfreq=512
        Datadict = super().__getitem__(index)
        signalwave=Datadict['data']
        rpeaks0=ecg.christov_segmenter(signalwave,sfreq)
        rpeaks1=ecg.christov_segmenter(-1*signalwave,sfreq)
        if sum(signalwave[rpeaks0[0]])<sum(-1*signalwave[rpeaks1[0]]):
            signalwave=-1*signalwave
        result=ecg.ecg(signalwave,sfreq)
        result_dict=result.as_dict()
        return result_dict['filtered'],Datadict['KSS']
#Face end to end Dataset

    
