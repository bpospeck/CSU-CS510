# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 17:05:32 2018

@author: Bradley
Mouse callback stuff based on this:
https://docs.opencv.org/3.4.1/db/d5b/tutorial_py_mouse_handling.html
"""
import numpy as np
import cv2
#global variables
makeBox = False # True when mouse pressed
boxDrawn = False # Always true after 1st box drawn
xCenter, yCenter = -1, -1
size = 96 #box is 96*2 x 96*2; Extends 96 around mouse click
init=False # Initialized on 1 frame

#Mouse callback fcn to draw bounding rectangle centered on mouse click
def drawRectangle(event,x,y,flags,param):
    global xCenter,yCenter,makeBox,boxDrawn,size,init
    if event == cv2.EVENT_LBUTTONDOWN:
        makeBox = True
        xCenter, yCenter = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        makeBox = False
        init = False
        boxDrawn = True
        cv2.rectangle(frame,(xCenter-size,yCenter-size),(xCenter+size,yCenter+size),(128,0,128),3)

if __name__=='__main__' :    
    tracker = cv2.TrackerKCF_create()              #Create kcf tracker
    cap = cv2.VideoCapture(0)                      #Initialize video capture
    cv2.namedWindow('original')                    #Setup different windows
    cv2.setMouseCallback('original',drawRectangle) #Setup mouse callback
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        print(len(gray.shape))
        # Display the resulting frame
        if boxDrawn:
            if not init:
                boundBox = (xCenter-size,yCenter-size,xCenter+size,yCenter+size)
                tracker.init(gray,boundBox)
                init = True
            else:
                success, boundBox = tracker.update(gray)
                if success:
                    upperLeft = (int(boundBox[0]),int(boundBox[1]))
                    lowerRight = (int(boundBox[0]+boundBox[2]),int(boundBox[1]+boundBox[3]))
                    cv2.rectangle(frame,upperLeft,lowerRight,(128,0,128),3)
        cv2.imshow('original',frame)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break      
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
