from mtcnn.mtcnn import MTCNN
import cv2
from utils import utils

ben_image = cv2.imread('Ben_affleck/ben1.jpg')
#cv2.imshow("mycastle", ben_image)

detector = MTCNN()

faces = detector.detect_faces(ben_image)
for face in faces:
    print(face)

marked_image = utils.create_bbox(ben_image)
filename = "saved_detections/ben_affleck_detection.jpg"
cv2.imwrite(filename, marked_image)

marked_image1 = utils.create_bbox(cv2.imread('Robert_jr/robert1.webp'))
filename = "saved_detections/robert_detection.jpg"
cv2.imwrite(filename, marked_image1)
