
note
1. this project is modified form https://github.com/nalbert9/Facial-Keypoint-Detection.git

2. the basic model is changed to Resnet to test the position-based method

3. in the future, will add heat-map base method and change to abstract class architecture
   to compare networks in different datasets. 

usage:
    0.the data can be download form the data folder form
         https://github.com/nalbert9/Facial-Keypoint-Detection.git
    1. create a python virtual environment
    2. run train.py to generate the model.pt file
    3. pick an image named as "tester1.jpg" and saved in the "images" foolder
    4. run test.py to test the face detection result by indicating the image name.     