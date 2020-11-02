#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2


# In[8]:


cascade_src="E:/New folder (2)/car haar cascade/cars.xml"


# In[9]:


video_src = "C:/Users/anush/Downloads/video.avi"


# In[10]:


import numpy as np
from matplotlib import pyplot as plt


# In[11]:


cap = cv2.VideoCapture(video_src)

car_cascade = cv2.CascadeClassifier(cascade_src)


# In[12]:


while True:
    ret, img = cap.read()
   
    if (type(img) == type(None)):
        break
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #img = cv.imread(gray,1)
    gaus = cv2.GaussianBlur(gray,(5,5),0)
    #img = cv.imread(gaus)
    
    plt.show()
    #cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    cars = car_cascade.detectMultiScale(gaus, 1.1, 2)


    for (x,y,w,h) in cars:
        cv2.rectangle(gaus,(x,y),(x+w,y+h),(0,255,255),2)
    
    cv2.imshow('Detecting...',gaus)
   
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
cap.release()


# In[ ]:




