# Import Necessary Packages
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

def create_bbox(image):
    
    '''
    This function detects the images, draws bounding boxes,
    and mark circle wherever the keypoints are there 
    '''


    detector = MTCNN()
    faces = detector.detect_faces(image)
    bounding_box = faces[0]['box']
    keypoints = faces[0]['keypoints']

    cv2.rectangle(image,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                (0,155,255),
                2)

    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

    return image


def extractFace(image,resize=(224,224)):
    
    '''
    This function reads the images and resize the face part .
    '''

    image = cv2.imread(image)
    detector = MTCNN()
    faces = detector.detect_faces(image)
    x1,y1,width,height = faces[0]['box']
    x2,y2 = x1 + width , y1 + height

    face_boundary = image[y1:y2,x1:x2] 
    face_image = cv2.resize(face_boundary,resize)
    return face_image


def getEmbedding(faces):
    
    '''
    This function get the faces and return the embeddings vector
    from the faces using VGGFace which is pretrained model that is being trained on 3.3 millions faces.

    '''

    face = np.asarray(faces,"float32")
    face = preprocess_input(face,version = 2)

    model = VGGFace(model = 'resnet50',include_top=False,input_shape=(224,224,3), pooling='avg')

    return model.predict(face)


def getSimilarity(faces):

    '''
    Get the Embedding from the faces and return the similarity score ,
    If it matches it returns a score less than 0.5 , else score above .5 
    '''

    embeddings = getEmbedding(faces)

    score = cosine(embeddings[0],embeddings[1])

    if score <=0.5 :
        return "Face Matched", score
    else:
        return "Face Doesnt Match", score

