import numpy as np
import cv2

SEQ_LEN = 75
NFRAMES = 9
SAMPLE_LEN = SEQ_LEN-NFRAMES   #Aathira : changed the equation
MAX_FRAMES = SAMPLE_LEN
MARGIN = NFRAMES/2

CHANNELS = 9
FRAME_ROWS = 128
FRAME_COLS = 128
COLORS = 1
LPC_ORDER = 8
NET_OUT = int(1/OVERLAP)*(LPC_ORDER+1)


def process_video(viddata, vidctr):
	        

def main():
    model = train.build_model(NET_OUT)
	model.compile(loss='mse', optimizer='adam')
	model.load_weights(weight_path)    
    cap = cv2.VideoCapture(0)
	faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+CASC_PATH)
	
    viddata = np.zeros((MAX_FRAMES,CHANNELS,FRAME_ROWS,FRAME_COLS),dtype="uint8")
    vidctr = 0
	temp_frames = np.zeros((SEQ_LEN*CHANNELS,FRAME_ROWS,FRAME_COLS),dtype="uint8")
	i = 0
    
    while True:
		ret,frame = cap.read()
		if ret==0:
			break
		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
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
		face = cv2.resize(face,(FRAME_COLS,FRAME_ROWS))
		face = np.expand_dims(face,axis=2)
		face = face.transpose(2,0,1)
		temp_frames[i*COLORS:i*COLORS+COLORS,:,:] = face
        i+=1
        if(i==75):
            for j in np.arange(MARGIN,SAMPLE_LEN+MARGIN):
		        viddata[vidctr,:,:,:] = temp_frames[COLORS*int(j-MARGIN):COLORS*int(j+MARGIN),:,:]
		        vidctr = vidctr+1
            Y_pred = model.predict(model, viddata)
            
            viddata = np.zeros((MAX_FRAMES,CHANNELS,FRAME_ROWS,FRAME_COLS),dtype="uint8")
            vidctr = 0
            temp_frames = np.zeros((SEQ_LEN*CHANNELS,FRAME_ROWS,FRAME_COLS),dtype="uint8")
            i = 0

if __name__ == "__main__":
	main()
