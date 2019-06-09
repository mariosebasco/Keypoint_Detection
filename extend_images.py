#!/usr/bin/env python3

# file made to flip images about the horizontal in order to get more training data
# as appears in research paper

import os
import numpy as np
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd


csv_file='data/training_frames_keypoints.csv'
key_pts_frame = pd.read_csv(csv_file)
skip_image = False

num_images = 3462

for idx in range(num_images):
    
    #read in original version
    image_name = os.path.join('data/training',
                              key_pts_frame.iloc[idx, 0])        
    
    image = mpimg.imread(image_name)

    
    if(image.shape[2] == 4):
        image = image[:,:,0:3]
        
    key_pts = key_pts_frame.iloc[idx, 1:].as_matrix()
    key_pts = key_pts.astype('float').reshape(-1, 2)

    # plt.imshow(image)
    # plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='g')
    # plt.show()

    # create flipped versions

    width, height, depth = image.shape
    zero_mat = np.zeros((width, height))

    for i in range(68):
        x = key_pts[i][0]
        y = key_pts[i][1]

        if(x > height):
            skip_image = True
            break
        if(y > width):
            skip_image = True
            break
        
        zero_mat[int(y)][int(x)] = 1

    if(skip_image):
        skip_image = False
        print("skipping image")
        continue
        
    zero_mat = np.fliplr(zero_mat)
    key_pts_flipped = np.zeros((68, 2))
    count = 0
    for i in range(width):
        for j in range(height):
            if zero_mat[i][j] == 1:
                key_pts_flipped[count][0] = float(j)
                key_pts_flipped[count][1] = float(i)
                count = count + 1


    # plt.imshow(image_flipped)
    # plt.scatter(key_pts_flipped[:, 0], key_pts_flipped[:, 1], s=20, marker='.', c='g')
    # plt.show()

    #write back flipped version
    f= open("data/training_frames_keypoints.csv","a+")
    f.write("\n")
    image_name = 'zz_' + repr(idx)  + '.jpg'
    f.write(image_name)
    for i in range(68):
        string = "," + repr(key_pts_flipped[i][0]) + "," + repr(key_pts_flipped[i][1])
        f.write(string)
    f.close()

    image_name = 'data/training/zz_' + repr(idx)  + '.jpg'
    image_flipped = np.fliplr(image)
    cv2.imwrite(image_name,image_flipped)

