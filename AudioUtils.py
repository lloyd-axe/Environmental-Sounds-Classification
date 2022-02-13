import random
import numpy as np
import pandas as pd
import librosa, librosa.display
import matplotlib.pyplot as plt

class AudioUtils():
    @staticmethod
    def open(audio_file, sr = 44100):
        sig, sr = librosa.load(audio_file, sr = sr)
        return (sig, sr)

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud
        if len(sig.shape) == 1 or sig.shape[0] == new_channel:
            return aud #do nothing          
        
        if new_channel == 1:
            resig = sig[:1, :]
        else:
            resig = np.concatenate([sig,sig])
        return (resig, sr)

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud
        resamp = librosa.resample(sig, orig_sr=sr, target_sr=newsr)
        if sr == newsr:
            return aud #do nothing
        return (resamp, newsr)

        #Resize to fixed length in milliseconds
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        sig_len = sig.shape[0]
        max_len = sr//1000 * max_ms
        
        if sig_len > max_len:
            sig = sig[:max_len]
            
        elif sig_len < max_len:
            #pad beginning and end
            pad_beg_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_beg_len
            
            #pad with 0s
            pad_beg = np.zeros(pad_beg_len)
            pad_end = np.zeros(pad_end_len)
            
            sig = np.concatenate((pad_beg, sig, pad_end))
            
        return (sig, sr)

    @staticmethod
    def spectrum(aud, display = False):
        sig, sr = aud
        sig_len = len(sig)
        frequency = np.fft.rfftfreq(sig_len, d=1/sr)
        magnitude = abs(np.fft.rfft(sig)/sig_len)
        if display:
            plt.plot(frequency, magnitude)
        return frequency, magnitude

    @staticmethod
    def spectrogram(
        aud, 
        log = False,
        hop_length = 512,
        n_fft = 1024, 
        display = False):
        sig, sr = aud
        #short time fourier transform
        stft = librosa.core.stft(sig, hop_length=hop_length, n_fft=n_fft)
        spectrogram = np.abs(stft)
        if log: 
            spectrogram = librosa.amplitude_to_db(spectrogram)
        if display:
            librosa.display.specshow(spectrogram, sr = sr, hop_length = hop_length)
            plt.colorbar()
            plt.show()
        return spectrogram
    
    @staticmethod
    def mfcc(
        aud, 
        n_mfcc = 13,
        hop_length = 512,
        delta = 0,
        display = False):
        sig, sr = aud
        mfccs = librosa.feature.mfcc(y=sig, sr= sr, n_mfcc=13)
        
        if delta > 0:
            mfccs = librosa.feature.delta(mfccs, order=delta)
        if display:
            librosa.display.specshow(mfccs, sr = sr, hop_length = hop_length)
            plt.colorbar()
            plt.show()
        return mfccs
    
    @staticmethod
    def envelope(aud, thresold = 0.0005):
        sig, sr = aud
        mask = []
        y = pd.Series(sig).apply(np.abs)
        y_mean = y.rolling(window = int(sr/10), min_periods=1, center = True).mean()
        for mean in y_mean:
            if mean > thresold:
                mask.append(True)
            else:
                mask.append(False)
        return (sig[mask], sr)

     #TimeShift
    @staticmethod
    def time_shift(aud, shift_lim):
        sig, sr = aud
        sig_len = sig.shape[0]
        shift_amt = int(random.random() * shift_lim * sig_len)
        return (np.roll(sig,shift_amt), sr)

    @staticmethod
    def stretch(sig, factor, n_fft= 1024):
        stft = librosa.core.stft(sig, n_fft=n_fft).transpose()  # i prefer time-major fashion, so transpose
        stft_rows, stft_cols = stft.shape

        times = np.arange(0, stft_rows, factor)  # times at which new FFT to be calculated
        hop = n_fft/4# frame shift
        stft_new = np.zeros((len(times), stft_cols), dtype=np.complex_)
        phase_adv = (2 * np.pi * hop * np.arange(0, stft_cols))/ n_fft
        phase = np.angle(stft[0])

        stft = np.concatenate( (stft, np.zeros((1, stft_cols))), axis=0)

        for i, time in enumerate(times):
            left_frame = int(np.floor(time))
            local_frames = stft[[left_frame, left_frame + 1], :]
            right_wt = time - np.floor(time)                        # weight on right frame out of 2
            local_mag = (1 - right_wt) * np.absolute(local_frames[0, :]) + right_wt * np.absolute(local_frames[1, :])
            local_dphi = np.angle(local_frames[1, :]) - np.angle(local_frames[0, :]) - phase_adv
            local_dphi = local_dphi - 2 * np.pi * np.floor(local_dphi/(2 * np.pi))
            stft_new[i, :] =  local_mag * np.exp(phase*1j)
            phase += local_dphi + phase_adv

        return librosa.core.istft(stft_new.transpose())

        
    
    