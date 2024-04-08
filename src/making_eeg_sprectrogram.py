import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt, gc
import pywt
import librosa


train = pd.read_csv('../data/raw/train.csv')

NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

directory_path = '../data/processed/EEG_Spectrograms/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

USE_WAVELET = None 


# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1):    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode='per')
    
    return ret


def spectrogram_from_eeg(parquet_path, eeg_min, eeg_max, display=False):    
    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = 200*(eeg_min+eeg_max)//2 ## 수정 
    middle = int(middle)
    eeg = eeg.iloc[middle:middle+10_000]
    
    # VARIABLE TO HOLD SPECTROGRAM
    # img = np.zeros((100,300,4),dtype='float32')
    img = np.zeros((128,256,4),dtype='float32') ## 수정 row x column x dim 

    if display: plt.figure(figsize=(10,7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]
        
        for kk in range(4):
            # FILL NANS
            x1 = eeg[COLS[kk]].values
            x2 = eeg[COLS[kk+1]].values
            m = np.nanmean(x1)
            if np.isnan(x1).mean()<1: x1 = np.nan_to_num(x1,nan=m)
            else: x1[:] = 0
            m = np.nanmean(x2)
            if np.isnan(x2).mean()<1: x2 = np.nan_to_num(x2,nan=m)
            else: x2[:] = 0
                
            # COMPUTE PAIR DIFFERENCES
            x = x1 - x2

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, 
                  hop_length=len(x)//256, 
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)
            ## 수정 hop_legnth = len(x)//이미지column수, n_mels= row수
            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32 ## 수정 width 30 ->32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]
            img[:,:,k] += mel_spec_db
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
                    
    return img


# Get all eegs with nans
PATH = '../data/raw/train_eegs/'
files = os.listdir(PATH)
eeg_nans = {}
for file in files:
    eeg = pd.read_parquet(f'{PATH}{file}').values
    if np.isnan(eeg).sum():
        eeg_nans[file] = np.isnan(eeg).sum()
np.save('../data/processed/eeg_nans', eeg_nans)

# 데이터 프레임 불러오기
df = pd.read_csv('~/kaggle-hms/data/processed/train-processed-addkey.csv')

# 자잘한 설정
PATH = '../data/raw/train_eegs/' ## 수정
DISPLAY = 4
EEG_IDS = df.eeg_id ## 수정
KEYS = df.key
EEG_MINS = df.eeg_min
EEG_MAXS = df.eeg_max

# 스펙트로그램 제작 시작
all_eegs = {}

for i, (eeg_id, key, eeg_min, eeg_max) in enumerate(zip(EEG_IDS,KEYS,EEG_MINS,EEG_MAXS)):
    if (i%100==0)&(i!=0): print(i,', ',end='')
    
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f'{PATH}{eeg_id}.parquet', eeg_min=eeg_min, eeg_max=eeg_max, display=False)
    
    # SAVE TO DISK
    if i==DISPLAY:
        print(f'Creating and writing {len(EEG_IDS)} spectrograms to disk... ',end='')
    np.save(f'{directory_path}{key}',img)
    all_eegs[key] = img
   
# SAVE EEG SPECTROGRAM DICTIONARY
np.save('../data/processed/eeg_specs', all_eegs)
