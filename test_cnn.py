#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
from models import Net


cap = cv2.VideoCapture(0)

net = Net()

#load the best saved model parameters (by your path name)
net.load_state_dict(torch.load('saved_models/50_epochs_128_batch.pt'))
    
# print out your net and prepare it for testing (uncomment the line below)
net.eval()


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # switch red and blue color channels 
    # --> by default OpenCV assumes BLUE comes first, not RED as in many images
    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = frame.copy()

    # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = face_cascade.detectMultiScale(image, 1.2, 2)

    # make a copy of the original image to plot detections on
    image_with_detections = image.copy()
    
    # loop over the detected faces, mark the image where each face is found
    for (x,y,w,h) in faces:
        # draw a rectangle around each detected face
        # you may also need to change the width of the rectangle drawn depending on image resolution
        cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)

        # Select the region of interest that is the face in the image 
        padding = 40 #show more of the face
        roi = image_with_detections[y:y+h + padding, x:x+w + padding]
    
        gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        norm_img = gray_img / 255.0
    
        norm_img = cv2.resize(norm_img, (224, 224))
    
        tensor_img = torch.from_numpy(norm_img.reshape(1,1,224,224))
        tensor_img = tensor_img.type(torch.FloatTensor)
    
        output_pts = net(tensor_img)
    
        torch.squeeze(output_pts)             # remove batch dim
        output_pts = output_pts.view(68, -1)
        predicted_key_pts = output_pts.data.numpy()

        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*55.0#+100
        predicted_key_pts[:, 0] = predicted_key_pts[:, 0] + x + w*0.5
        predicted_key_pts[:, 1] = predicted_key_pts[:, 1] + y + h*0.5
        
        for i in range(68):
            pt1_x = predicted_key_pts[i][0]
            pt1_y = predicted_key_pts[i][1]

            # pt2_x = predicted_key_pts[i + 1][0]
            # pt2_y = predicted_key_pts[i + 1][1]

            # cv2.line(image_with_detections,(pt1_x,pt1_y),(pt2_x,pt2_y),(255,0,0),1)
            cv2.circle(image_with_detections, (pt1_x, pt1_y), 5, (255, 0, 0), -1)
            
    # Display the resulting frame
    cv2.imshow('frame',image_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# # load in color image for face detection
# image = cv2.imread('images/obamas.jpg')


# # fig = plt.figure(figsize=(9,9))
# # plt.imshow(image_with_detections)
# # plt.show()

##########################################################################################


# image_copy = np.copy(image)

# # loop over the detected faces from your haar cascade
# counter = 0

# for (x,y,w,h) in faces:
    

#     print(predicted_key_pts)

    
#     plt.imshow(gray_img, cmap='gray')
#     plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=60, marker='.', c='m')
    
#     break
    
# plt.show()
