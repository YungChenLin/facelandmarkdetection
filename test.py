import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torch import nn
from torch.autograd import Variable
import models.base_model as modelib
import cv2

#from torchvision import transforms, utils
from data_load import Rescale, RandomCrop, Normalize, ToTensor

def show_face_detection(faces, image_with_detections):
    for (x,y,w,h) in faces:
        # draw a rectangle around each detected face
        # you may also need to change the width of the rectangle drawn depending on image resolution
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),2)
        # cv2.imshow("show", image_with_detections)# opencv use BGR 
        # cv2.waitKey(500)
        # cv2.destroyWindow("show")
    fig = plt.figure(figsize=(9,9))
    plt.imshow(image_with_detections, cmap='gray')

def show_all_keypoints(images, keypoints):  
    batch_size = len(images)
    plt.figure(figsize=(50, 5))
    for i, face in enumerate(images):
        ax = plt.subplot(1, batch_size, i+1)
        # un-transform the predicted key_pts data
        predicted_keypoints = keypoints[i].data
        predicted_keypoints = predicted_keypoints.numpy()
        plt.imshow(face, cmap='gray')
        plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], s=20, marker='.', c='m')
        #plt.axis('off')
    plt.show()

## load in color image for face detection
#image = cv2.imread('images/obamas.jpg', 1)
image = cv2.imread('images/tester2.jpg', 1)
#image = cv2.resize(image, (800,600))
cv2.imshow("show", image)
cv2.waitKey(500)
cv2.destroyWindow("show")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# switch red and blue color channels 
# --> by default OpenCV assumes BLUE comes first, not RED as in many images
#gray = image.copy()
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot the image
fig = plt.figure(figsize=(9,9))
plt.imshow(image)
 #-----------------------face detection -------------------------------------------------------------------   
# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_smile.xml')

# load in a haar cascade classifier for detecting eyes
eye_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_eye.xml')

# run the detector
# the output here is an array of detections; the corners of each detection box
# if necessary, modify these parameters until you successfully identify every face in a given image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()
show_face_detection(faces, image_with_detections)

net = modelib.Net()
## load the best saved model parameters 
net.load_state_dict(torch.load('./saved_models/facial_keypoints_model_res1.pt'))

## print out your net and prepare it for testing (uncomment the line below)
net.eval()
PADDING = 40
images = []
keypoints = []
image_copy = np.copy(image)
#print(len(faces))
# loop over the detected faces from your haar cascade
for (x,y,w,h) in faces:
    # Select the region of interest that is the face in the image 
    roi = image_copy[y-PADDING:y+h+PADDING, x-PADDING:x+w+PADDING,:]
    ## Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi = (roi / 255.).astype(np.float32)
    ## Rescale the detected face to be the expected square size for CNN (224x224, suggested)
    roi = cv2.resize(roi, (224, 224))
    images.append(roi)
    
    ## Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    if len(roi.shape) == 2:
        roi = np.expand_dims(roi, axis=0)
    else:
        roi = np.rollaxis(roi, 2, 0)
    
    # Match the convolution dimensions
    roi = np.expand_dims(roi, axis=0)
    
    ## Make facial keypoint predictions using your loaded, trained network 
    # Forward pass
    roi = torch.from_numpy(roi).type(torch.FloatTensor)
    output_pts = net.forward(roi)
    
    output_pts = output_pts.view(output_pts.size()[0], 68, -1)
    output_pts = output_pts*112 + 112

    keypoints.append(output_pts[0])

cv2.destroyAllWindows()

show_all_keypoints(images, keypoints)