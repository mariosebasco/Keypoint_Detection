#!/usr/bin/env python3

import os
import numpy as np
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

#regular
idx = 131
#flipped
idx_flipped = idx + 3462

csv_file='data/training_frames_keypoints.csv'
key_pts_frame = pd.read_csv(csv_file)

#non flipped
image_name = os.path.join('data/training',
                          key_pts_frame.iloc[idx, 0])        

image = mpimg.imread(image_name)

if(image.shape[2] == 4):
    image = image[:,:,0:3]
        
key_pts = key_pts_frame.iloc[idx, 1:].as_matrix()
key_pts = key_pts.astype('float').reshape(-1, 2)

plt.imshow(image)
plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='g')
plt.show()


#flipped
image_name = os.path.join('data/training',
                          key_pts_frame.iloc[idx_flipped, 0])        

image = mpimg.imread(image_name)

if(image.shape[2] == 4):
    image = image[:,:,0:3]
        
key_pts = key_pts_frame.iloc[idx_flipped, 1:].as_matrix()
key_pts = key_pts.astype('float').reshape(-1, 2)

plt.imshow(image)
plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='g')
plt.show()

