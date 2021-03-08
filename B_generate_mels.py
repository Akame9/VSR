import B_audio
from b_hparams import hparams as hp
import numpy as np
from os import listdir, path
import os
import librosa
from scipy.io.wavfile import write

def generating_mels(file,fulldir):
    list=[0.0,0.4,0.8,1.2,1.6,2.0]
    all_mels=[]
    all_arrays=[]
    for i in range(0,len(list)):
        wavpath="C:/Project/VSR/B_dataset/wavs/"+str(file[29:35])+"/"+str(list[i])+"-1.wav"
        wav = B_audio.load_wav(wavpath, hp["sample_rate"])
        print(wav.shape)
        print(type(wav))
        spec = B_audio.melspectrogram(wav, hp)
        #lspec = B_audio.linearspectrogram(wav, hp)
        print("type",type(spec))
        print("shape",spec.shape)
        all_mels.append(np.transpose(spec))
        print("shape of all mels",all_mels[i].shape)
        if (i==5):
            padded = np.zeros(all_mels[4].shape)
            padded[:all_mels[5].shape[0]] = all_mels[5]
            all_mels[5]=padded
            print("shape after appending", all_mels[5].shape)
        

        #reading all mels to create wavs for cross checking    
        data= np.transpose(all_mels[i])
        print(data.shape)
        wav = B_audio.inv_mel_spectrogram(data,hp)
        B_audio.save_wav(wav,'C:/Project/VSR/check_mels'+str(i)+'.wav' , sr=16000)
        
    #appending all files into 1 file
    for i in range (0,len(all_mels)):
        all_arrays.append(np.transpose(all_mels[i]))
            
    print("shape of all 6",np.shape(all_arrays))
    np.save(fulldir+"all_6", all_arrays)

        
def main():
    file='C:/Project/VSR/B_dataset/s2/bbaf1n/audio.wav'
    fulldir="C:/Project/VSR/B_dataset/all_in_one/"+str(file[29:35])
    os.makedirs(fulldir, exist_ok=True)
    generating_mels(file,fulldir)


if __name__ == '__main__':
	main()
