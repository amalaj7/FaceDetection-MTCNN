# Face Detection and Identification
----------------------------------------

- Used MTCNN for Face detection
- Extracted the Facial Features using VGGFace(resnet50 model)
- Used CosineDistance to Identify the Facial Features and return the similarity score .

## What is MTCNN? 

- MTCNN is a state of art model for Face detection and various other stuffs.
- Multi-task Cascaded Convolutional Networks (MTCNN)
- It is known as MultiTasking because it not only detects the face , it returns the BBox Regression as well as Keypoints coordinates in the faces like Left-Eye,Right-eye,Nose,Mouth
- Research Paper : [MTCNN Paper](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)
- Github Repo : [Github](https://github.com/ipazc/mtcnn)


## How to Run 


For Face Detection run :
```bash
python face_detection.py
```

For Face Identification run:
```bash
python face_identify.py
```