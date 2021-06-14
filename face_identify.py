from mtcnn.mtcnn import MTCNN
import cv2 as cv2
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from utils import utils

ben_image = cv2.imread('Ben_affleck/ben1.jpg')

detector = MTCNN()

faces = detector.detect_faces(ben_image)
for face in faces:
    print(face)

#marked_image = utils.create_bbox(ben_image)

extract_face = utils.extractFace('Robert_jr/robert1.webp')
filename = "Extracted_FaceImage/robert_cropped.jpg"
cv2.imwrite(filename, extract_face)

# To extract the face and get the embeddings
extractedFaces = [utils.extractFace(image) for image in ['Ben_affleck/ben1.jpg','Robert_jr/robert1.webp']]

# To compare the faces and get the similarity score 
print(utils.getSimilarity(extractedFaces))