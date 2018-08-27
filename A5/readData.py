# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:00:24 2018

@author: Bradley
"""

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
    if np.array_equal(data[:64*30],data[64*30:64*60]):
        print("EQUal")
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
        
if __name__ == '__main__' :
    n,labels,objs = readObjs()
    print('num objs: {0}'.format(n))
    for i in range(n):
        print('num frames for {0}: {1}'.format(labels[i],len(objs[i])))