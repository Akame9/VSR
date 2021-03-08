import subprocess
from glob import glob
import os
from os import path



#ffmpeg to cut audio frames
#from 0 sec go ahead 1 sec
#ffmpeg -ss 0 -t 1 -i C:/Users/BHAVANA/Desktop/audio.wav C:/Users/BHAVANA/Desktop/frame0-0.1.wav
#ffmpeg -ss 0.4 -t 1 -i C:/Users/BHAVANA/Desktop/audio.wav C:/Users/BHAVANA/Desktop/frame0-0.1.wav
#ffmpeg -ss 0.8 -t 1 -i C:/Users/BHAVANA/Desktop/audio.wav C:/Users/BHAVANA/Desktop/frame0-0.1.wav

file='C:/Project/VSR/B_dataset/s2/bbaf1n/audio.wav'
print(file.shape)
template = 'ffmpeg -ss {} -t 1 -i {} {} '

fulldir='C:/Project/VSR/B_dataset/wavs/'+str(file[29:35])
os.makedirs(fulldir, exist_ok=True)

def main():
    list=[0.0,0.4,0.8,1.2,1.6,2.0]
    for i in range(0,len(list)):
        command = template.format(list[i],file,fulldir+"/"+str(list[i])+"-1.wav")
        subprocess.call(command, shell=True)


if __name__ == '__main__':
	main()