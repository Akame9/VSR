import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import sys
from os import path
import audio_tools as aud
import audio_read as ar


FRAME_ROWS = 96
FRAME_COLS = 96
SEQ_LEN = 75
DEFAULT_FACE_DIM = 292
CASC_PATH = 'haarcascade_frontalface_alt.xml'
DATAPATH = r'C:/Project/VSR/s2/video/mpg_6000'
ALL_FRAMES=[]

def process_video(vf, faceCascade):
	frames=[]
	fulldir='C:/Project/VSR/B_dataset/'
	cap = cv2.VideoCapture(vf)
	for i in np.arange(SEQ_LEN):
		ret,frame = cap.read()
		if ret==0:
			break
		faces = faceCascade.detectMultiScale(
			frame,
			scaleFactor=1.05,
			minNeighbors=3,
			minSize=(200,200),
			flags = 2
		)
		if len(faces)==0 or len(faces)>1:
			print ('Face detection error in %s frame: %d'%(vf, i))							
			face = frame[199:-85,214:-214] # hard-coded face location
		else:
			for (x,y,w,h) in faces:
				face = frame[y:y+DEFAULT_FACE_DIM,x:x+DEFAULT_FACE_DIM]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) 
		face = cv2.resize(face,(FRAME_ROWS,FRAME_COLS))
		frames.append(face)	
	counter=0
	for i in range(0,6):
		i=counter
		ALL_FRAMES.append(frames[i:i+25])
		counter+=10
	
	
				

def main():
	print ('Processing video data...')
	faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+CASC_PATH)
	vidfiles = [f for f in listdir(DATAPATH) if isfile(join(DATAPATH, f)) and f.endswith(".mpg")]
	counter=0
	for i in range(5):
		global ALL_FRAMES	
		i=counter
		sets=vidfiles[i:i+200]
		for vf in sets:
			process_video(join(DATAPATH,vf), faceCascade)
		counter+=200
		np.save("C:/Project/VSR/B_dataset/{}.npy".format(i+1), ALL_FRAMES)
		ALL_FRAMES=[]
		
		
	
			
	
if __name__ == "__main__":
	main()