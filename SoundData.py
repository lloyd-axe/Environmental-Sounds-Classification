import os
import numpy as np
from AudioUtil import AudioUtil

class SoundDS:
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.durations = 3000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4
        self.hop_length = 512
        self.n_fft = 1024
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        audio_file = os.path.join(self.data_path, os.path.normpath(self.df['relative_path'][idx]))
     
        class_id = self.df.loc[idx, 'classID']
        
        aud = AudioUtil.open(audio_file)
        aud = AudioUtil.pad_trunc((AudioUtil.stretch(aud[0], 0.84), aud[1]), 3000)
        #spec = AudioUtil.spectrogram(aud, hop_length=self.hop_length, n_fft=self.n_fft)
        #log_spec = AudioUtil.spectrogram(aud, hop_length=self.hop_length, n_fft=self.n_fft, log = True)
        mfcc = AudioUtil.mfcc(aud)
        mfcc = mfcc.T
        return mfcc, class_id