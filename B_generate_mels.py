#generating mel
import B_audio
from b_hparams import hparams as hp
import numpy as np
from os import listdir, path
import librosa
from scipy.io.wavfile import write


#ffmpeg to cut audio frames
#from 0 sec go ahead 1 sec
#ffmpeg -ss 0 -t 1 -i C:/Users/BHAVANA/Desktop/audio.wav C:/Users/BHAVANA/Desktop/frame0-0.1.wav
#ffmpeg -ss 0.4 -t 1 -i C:/Users/BHAVANA/Desktop/audio.wav C:/Users/BHAVANA/Desktop/frame0-0.1.wav
#ffmpeg -ss 0.8 -t 1 -i C:/Users/BHAVANA/Desktop/audio.wav C:/Users/BHAVANA/Desktop/frame0-0.1.wav

#generating mel using B_audio functions
wavpath="C:/Project/GRID_Data/frame1.wav"
wav = B_audio.load_wav(wavpath, hp["sample_rate"])
print(wav.shape)
print(type(wav))
fulldir="C:/Users/Bhavana/Desktop"
specpath = path.join(fulldir, 'frame1.npz')
spec = B_audio.melspectrogram(wav, hp)
lspec = B_audio.linearspectrogram(wav, hp)
np.savez_compressed(specpath, spec=spec, lspec=lspec)




#Load the generated 6 mels with overlap of 15 frames
data_1 = np.load('C:/Users/BHAVANA/Desktop/mels_0-1.npz')['spec'].T
data_2 = np.load('C:/Users/BHAVANA/Desktop/mels_0.4-1.npz')['spec'].T
data_3 = np.load('C:/Users/BHAVANA/Desktop/mels_0.8-1.npz')['spec'].T
data_4 = np.load('C:/Users/BHAVANA/Desktop/mels_1.2-1.npz')['spec'].T
data_5 = np.load('C:/Users/BHAVANA/Desktop/mels_1.6-1.npz')['spec'].T
data_6 = np.load('C:/Users/BHAVANA/Desktop/mels_2.0-1.npz')['spec'].T

'''print("1",data_1.shape)
print("2",data_2.shape)
print("3",data_3.shape)
print("4",data_4.shape)
print("5",data_5.shape)
print("6",data_6.shape)'''

#padding the last mel from (79,80) to (81,80)
padded = np.zeros(data_5.shape)
padded[:data_6.shape[0]] = data_6
data_6=padded
np.savez_compressed('C:/Users/BHAVANA/Desktop/mels_constructed.npz', spec=data_6) # save the padded data

#load the padded data to convert to wav for cross-checking
padded_data= np.load('C:/Users/BHAVANA/Desktop/mels_constructed.npz')['spec']
wav = B_audio.inv_mel_spectrogram(padded_data.T,hp)
B_audio.save_wav(wav,'C:/Project/VSR/code/check_mel.wav' , sr=16000)


# Concatenate 
np.savez_compressed('C:/Project/GRID_Data/set_1', a=data_1,b=data_2,c=data_3,d=data_4,e=data_5,f=data_6 )
loaded = np.load('C:/Project/GRID_Data/set_1.npz')
print(loaded.files)
