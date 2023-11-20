import numpy as np
import matplotlib.pyplot as plt
import torch
import pywt




##CWT
def getCWTImage(data,t,sfreq,totalscal = 512,sampling_rate=512,wavename = "cgau8"):
    '''
    输入：
        data:数据序列
        t:时间轴
        sampling_rate，sfreq:采样频率
        totalscal:最大时间尺度
        wavename：小波函数名
    返回 :
        t,时间轴
        frequencies:频率轴
        log_cwtmatr_uniform:能量系数

    https://blog.csdn.net/weixin_46713695/article/details/127234673
    '''
    #时间
    #print(data)
    # wavename = "cgau8"
    #wavename ="mexh"
    # totalscal = 512   # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
    # sampling_rate=512

    fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
    cparam = 2 * fc * totalscal  # 常数c
    scales = np.arange(8, totalscal/4, 1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
    [cwtmatr, frequencies] = pywt.cwt(data[0][0], scales, wavename, 1.0/sampling_rate)  # 连续小波变换模块
    log_cwtmatr=torch.log10(torch.tensor(abs(cwtmatr)))# 取对数结果

    # 归一化
    max_l=torch.max(log_cwtmatr.to(torch.float32))
    min_l=torch.min(log_cwtmatr.to(torch.float32))
    log_cwtmatr_uniform=(log_cwtmatr-min_l)/(max_l-min_l)#归一化的结果

    #uni_log_dataframe=pd.DataFrame(log_cwtmatr_uniform,columns=t,index=frequencies)

    return t,frequencies,log_cwtmatr_uniform
    pass
##CWT
def getCWTImage(data,t,sfreq,totalscal = 512,sampling_rate=512,wavename = "cgau8"):
    '''
    输入：
        data:数据序列
        t:时间轴
        sampling_rate，sfreq:采样频率
        totalscal:最大时间尺度
        wavename：小波函数名
    返回 :
        t,时间轴
        frequencies:频率轴
        log_cwtmatr_uniform:能量系数

    https://blog.csdn.net/weixin_46713695/article/details/127234673
    '''
    fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
    cparam = 2 * fc * totalscal  # 常数c
    scales = np.arange(8, totalscal/4, 1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
    [cwtmatr, frequencies] = pywt.cwt(data[0][0], scales, wavename, 1.0/sampling_rate)  # 连续小波变换模块
    log_cwtmatr=torch.log10(torch.tensor(abs(cwtmatr)))# 取对数结果
    # 归一化
    max_l=torch.max(log_cwtmatr.to(torch.float32))
    min_l=torch.min(log_cwtmatr.to(torch.float32))
    log_cwtmatr_uniform=(log_cwtmatr-min_l)/(max_l-min_l)#归一化的结果

    return t,frequencies,log_cwtmatr_uniform
    pass

##生成可验证数据
def CreateSignal(random=False,A=1,length=1):
    '''
    生成频率在0-20之间20个频率的合成信号
    输入：
        random=False时生成1-20等差数列
        random=True时生成0-20之间服从高斯分布的20个随机数
    返回：
        x:时间
        y:信号幅值
        power:应有总能量
    '''
    if random:
        fs=np.random.random((20,1))*20
    else:
        fs=np.linspace(1,20,20).reshape((20,1))
    x=np.linspace(0,length*1,length*512)
    y=A*np.sin(2*np.pi*x*fs)
    y=np.sum(y,axis=0)
    return x,y,A**2*length*0.5*20

def FFTParse(y,sfreq):
    '''
    对信号进行fft变换，生成双边序列
    输入：
        y:时间序列
        sfreq:采样频率
    输出：
        fft_result:幅值双边序列
        freq:频率双边序列
    '''
    fft_result=np.fft.fftshift(np.fft.fft(y))/len(y)
    freq=np.fft.fftshift(np.fft.fftfreq(len(y),1/sfreq))

    return fft_result,freq

def PlotAFR(fft_result,freq):
    '''
    绘制幅频响应
    输入：
        fft_result:幅值序列
        freq:频率序列
    '''
    
    plt.figure()
    plt.plot(freq,abs(fft_result))
    plt.title("linespace")
    plt.xlabel("freq")
    plt.ylabel("amplitude")
    plt.figure()
    plt.scatter(freq,abs(fft_result))
    plt.title("linespace")
    plt.xlabel("freq")
    plt.ylabel("amplitude")
    plt.figure()
    plt.scatter(freq,20*np.log10(abs(fft_result)))
    plt.title("logspace")
    plt.xlabel("freq")
    plt.ylabel("dB")
    plt.figure()
    plt.plot(freq,20*np.log10(abs(fft_result)))
    plt.title("logspace")
    plt.xlabel("freq")
    plt.ylabel("dB")

def BandPower(fft_result,freq,min,max):
    '''
    计算双边序列指定频带能量
    输入：
        fft_result:幅值双边序列
        freq:频率双序列
        min：频带下限
        max：频带上限
    输出：
        指定频带能量
    算法说明：
        际的能量是双边的能量，而如果直接将单边的幅值*2的话，所得能量是实际能量的二倍，所以计算了单边频谱能量*2
    '''
    dw=freq[1]-freq[0]
    idx_band=np.logical_and(min<=freq,freq<=max)
    #计算了单边的能量函数，而实际的能量是双边的能量，而如果直接将单边的幅值*2的话，所得能量是实际能量的二倍
    bandpower=sum(abs(fft_result[idx_band])**2*2)*dw
    
    return bandpower

def SignalBandPower(y,sfreq,min,max):
    '''
    输入：
        y：数据：list
        sfreq:采样频率
        min：频带下限
        max：频带上限
    输出：
        该频段的能量
    
    '''
    fft_result,freq=FFTParse(y,sfreq)
    bandpower=BandPower(fft_result,freq,min,max)
    return bandpower

def VerifyAlgorithm(min=0,max=100):
    '''
    验证FFT频域计算能量算法
    '''
    x,y,power=CreateSignal(random=False)
    fft_result,freq=FFTParse(y,512)
    PlotAFR(fft_result,freq)
    print(BandPower(fft_result,freq,min,max))



def SequenceFFT(y,sfreq,N_FFT=15*512,HOP_LENGTH=5*512):
    '''
    计算一段长数据的FFT
    输入：
        y:数据：list
        N_FFT,FFT运算序列长度即窗长
        HOP_LENGTH:窗平移步幅即步长
    输出：
        stft:FFT结果列表（双边），每一列是一一个FFT结果，每一行是一个频率随时间变换的序列
        freq:双边频率序列

    '''
    startIndex=torch.range(0,((len(y)-N_FFT)//HOP_LENGTH))*HOP_LENGTH
    endIndex=startIndex+N_FFT
    
    idxs=torch.stack((startIndex,endIndex),dim=1).int()
    stft=[]
    for idx in idxs :
        fft_result,freq=FFTParse(y[idx[0]:idx[1]],sfreq)
        stft.append(fft_result)       
    stft=np.array(stft).T
    return stft,freq


##STFT
