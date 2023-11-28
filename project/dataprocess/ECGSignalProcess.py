from biosppy.signals import ecg
import mne
def Calibration_direction(signal,sfreq):
    rpeaks0=ecg.hamilton_segmenter(signal.T,sfreq)
    rpeaks1=ecg.christov_segmenter(-1*signal,sfreq)
    if sum(signal[rpeaks0[0]])<sum(-1*signal[rpeaks1[0]]):
        signal=-1*signal
    return signal

def Denoising(signal,sfreq):
    noise=mne.filter.filter_data(signal,sfreq,l_freq=0,h_freq=0.01,verbose=False)
    signal=signal-noise
    return signal