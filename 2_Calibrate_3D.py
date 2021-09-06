#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

cap= cv.VideoCapture('C:/Users/HP/Desktop/Task3_Nihal/Videos/3D_metal_printer.h264')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv.imwrite('C:/Users/HP/Desktop/Task3_Nihal/frame_'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()


# In[15]:


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6,7), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
            
        # Draw and display the corners
        cv.drawChessboardCorners(img, (6,7), corners2, ret)
        cv.imwrite('Correction.png', img)
        plt.imshow(img)


# In[16]:


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera Calibration:\n ",ret)
print("\nCamera Matrix:\n ",mtx)
print("\nDistortion Parameters\n: ",dist)
print("\nRotation Vectors:\n ",rvecs)
print("\nTranslation Vectors:\n ",tvecs)


# In[17]:


path = glob.glob('*.jpg')
img = []
cnt = 0
for i in path:
    #print(i)
    n = cv.imread(i)
    cnt = cnt+1
    h,  w = n.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(n, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('result%d.jpg' %cnt, dst)


# In[18]:


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6,7), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
            
        # Draw and display the corners
        cv.drawChessboardCorners(img, (6,7), corners2, ret)
        cv.imwrite('Correction.png', img)
        plt.imshow(img)


# In[19]:


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera Calibration:\n ",ret)
print("\nCamera Matrix:\n ",mtx)
print("\nDistortion Parameters\n: ",dist)
print("\nRotation Vectors:\n ",rvecs)
print("\nTranslation Vectors:\n ",tvecs)


# In[20]:


from numpy.linalg import inv
inv(mtx)


# In[21]:


import math
import numpy as np
from numpy import linalg as LA
p1 = np.array([703, 475, 1])
p2 = np.array([834, 472, 1])
v1 = np.matmul(inv(mtx) , p1)
v2 = np.matmul(inv(mtx) , p2)
theta = math.acos(np.dot(v1,v2) / (LA.norm(v1) * LA.norm(v2)))
width_height = 2*(484*math.tan(theta/2))
distance = 25/math.tan(theta/2)
print(distance)


# In[ ]:


h1 = np.array([675, 240,1])
h2 = np.array([691, 774, 1])
v11 = np.matmul(inv(mtx) , h1)
v12 = np.matmul(inv(mtx) , h2)
theta1 = math.acos(np.dot(v11,v12) / (LA.norm(v11) * LA.norm(v12)))
height = 2*(376*math.tan(theta1/2))
print(height)

w1 = np.array([685, 472, 1])
w2 = np.array([960, 470, 1])
v21 = np.matmul(inv(mtx) , w1)
v22 = np.matmul(inv(mtx) , w2)
theta2 = math.acos(np.dot(v21,v22) / (LA.norm(v21) * LA.norm(v22)))
width = 2*(376*math.tan(theta2/2))
print(width)


# In[ ]:





# In[ ]:




