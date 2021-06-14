# Face Detection and Identification
----------------------------------------

- Used MTCNN for Face detection
- Extracted the Facial Features using VGGFace(resnet50 model)
- Used CosineDistance to Identify the Facial Features and return the similarity score .

For Face Detection run :
```bash
python detection.py
```

For Face Identification run:
```bash
python face_verification.py
```