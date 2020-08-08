import cv2
import os
import pickle
import face_recognition
from PIL import Image
import numpy as np 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'dataset')

knownEncodings = []
knownNames = []

for (root, dirs, files) in os.walk(image_dir):
	for file in files:
		if file.endswith('jpg') or file.endswith('png'):
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path)).replace(' ', '-').lower()
			#print(label)
			#pil_image = Image.open(path).convert('L')
			pil_image = cv2.imread(path)
			rgb = cv2.cvtColor(pil_image, cv2.COLOR_BGR2RGB)
			boxes = face_recognition.face_locations(rgb)
			encodings = face_recognition.face_encodings(rgb, boxes)
			for encoding in encodings:
				knownEncodings.append(encoding)
				knownNames.append(label)

print("[INFO] serializing encodings...")
data = {'encodings': knownEncodings, 'names':knownNames}
f = open('enen.txt', 'wb')
f.write(pickle.dumps(data))
f.close()





