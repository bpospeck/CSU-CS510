# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 21:53:46 2018

@author: Bradley
tracking code base from here:
https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
"""

import cv2
import sys
import numpy as np

def readObjs():
    file = open('objs.txt','r')
    header = file.readline()
    header = header.strip('\n')
    file.close()
    names = header.split(' ')
    numObjs = 0
    namesLs = []
    for name in names:
        numObjs +=1
        namesLs.append(name)
    data = np.genfromtxt('objs.txt',dtype=np.uint8,skip_header=1)
    objs = []
    maxFeats = 64
    numFrames = 30
    for n in range(numObjs):
        frame = []
        for f in range(numFrames):
            feat = []
            for m in range(maxFeats):
                feat.append(data[n*numFrames*maxFeats + f*maxFeats + m])
            frame.append(np.array(feat))
        objs.append(frame)
    return numObjs,namesLs,objs

def compareObj(desc,objs,bf):
    objSums = []
    for obj in objs:
        frameSums = []
        for f in obj:
            matches = bf.knnMatch(desc,f,k=2)
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    #print('m:{0} n:{1}'.format(m.distance,n.distance))
                    frameSums.append(m.distance)
        objSums.append(np.sum(np.array(frameSums)))
    print('king:{0} farseek:{1} lock:{2} sleeve:{3} KH:{4}\n'.format(objSums[0],objSums[1],objSums[2],objSums[3],objSums[4]))
    maxIndex = np.argmax(objSums)
    if objSums[maxIndex] < 1725:
        return -1
    else:
        return maxIndex

if __name__ == '__main__' :
    tracker= None #Declaring it, will be initialized along with ROI
    desc = None #will store initial ORB descriptors
    tracking = False
    newObj = False
    n,labels,objs = readObjs()
    #print(objs[0][0])
    #print(len(objs[0][0][0]))
    orb = cv2.ORB_create(nfeatures=64,edgeThreshold=5)
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck = False)
    video = cv2.VideoCapture(0)
    if not video.isOpened():#exit if video not opened
        print ("Could not open video")
        sys.exit()
    # Get first frame
    ok, frame = video.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit() 
    #Find camera center, and go to upper left corner of 256x256 tracking init box
    x = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)/2 - 96)
    y = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)/2 - 96)
    box = frame[x:x+192, y:y+192].copy()
    cv2.namedWindow("Tracking")
    cv2.namedWindow("Bound box")
    while True:
        ok, frame = video.read()
        #h = hog.compute(frame)
        #print(len(h))
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if not ok:
            break
        if not tracking:
            cv2.putText(frame, "Not tracking now",(100,20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
            bbox=(x,y,192,192)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            key = cv2.waitKey(20) & 0xFF
            if key==ord('t'): # 't'rack object
                tracking = True
                tracker = cv2.TrackerKCF_create()
                ok = tracker.init(frame,bbox)
                if ok:
                    box = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])].copy()
                    #cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                    _, desc = orb.detectAndCompute(box,None)
                    if newObj:
                        file = open("objs.txt",'a')
                        for i in range(30):
                            _, frame = video.read()
                            _, bbox = tracker.update(frame)
                            p1 = (int(bbox[0]), int(bbox[1]))
                            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                            box = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])].copy()
                            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                            _, desc = orb.detectAndCompute(box,None)
                            for de in desc:
                                for d in de:
                                    file.write(str(d))
                                    file.write(' ')
                                file.write('\n')
                            file.write('\n')
                        newObj = False
                        file.close()
        else:
            # Start timer
            timer = cv2.getTickCount()
            # Update tracker
            ok, bbox = tracker.update(frame)
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
            if ok:
                # Tracking success: Draw bounding box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                box = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])].copy()
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                _, desc = orb.detectAndCompute(box,None)
                nameIndex = compareObj(desc,objs,bf)
                if nameIndex == -1:
                    cv2.putText(frame,'Unknown', (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0),2)
                else:
                    cv2.putText(frame,labels[nameIndex], (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0),2)
                #matches = bf.match(desc,objs[0][0])
                #print(len(desc[0]))
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            # Display tracker type, fps, and result on frame
            cv2.putText(frame, "KCF Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
            #cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        cv2.imshow("Tracking", frame)
        cv2.imshow("Bound box", box)
        #if desc is not None:
        #    print(len(desc))
        # Exit if ESC pressed
        key = cv2.waitKey(1) & 0xff
        if key == 27 : 
            break
        elif key == ord('r'): # 'r'eset object it's tracking
            tracking = False
        elif key == ord('n'): # 'n'ew object to recognize
            tracking = False
            newObj = True
    video.release()
    cv2.destroyAllWindows()