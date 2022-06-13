# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:53:59 2021

@author: Stefani Dimitrova std31
"""

#import libraries

import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import operator
import csv
import pandas as pd
import time
from datetime import timedelta


emily = pd.read_csv(r'C:\Users\44736\Desktop\horseblink\data\Blink_rate_automation_GH10004.csv')

#global variables
ROIsize = 100 # area of interest size
BGRwinsize = 60 # framerate per second is 60
BGRwinoffset = int(BGRwinsize/2)-1
stopframe = int(input('Chose frame to stop at:'))
startframe = int(input('Chose frame to start at:'))

amp_threshold = -8
frame_threshold = 20
global totalframenumber
global frame_inner
fps = 60


# if the number of frames the user has set is less than 30 make it equal to 30, the program requires at least 30 frames to
# perform adequate analysis for the background
if stopframe < BGRwinsize:
    stopframe = BGRwinsize
    print('Error, stopframe is changed to:', stopframe)

#font specification
font = cv2.FONT_HERSHEY_SIMPLEX;
org = (50, 50);
fontScale = 1;
font_color = (255, 255, 0);
thickness = 2;


#create tracker
tracker = cv2.TrackerCSRT_create(); 
backup = cv2.TrackerCSRT_create();

#function that allows us to select file
def openvideofile():
#the root window into which all other widgets go. It is an instance of the class Tk
    root = tk.Tk()
#Removes the window from the screen (without destroying it)
    root.withdraw()

    file_path = filedialog.askopenfilename()
    return file_path


#Retrieve first frame where we select area of interest
def initialize_camera(cap):
    _, frame = cap.read()
    return frame

    box = cv2.selectROI(frame, False);
    tracker.init(frame, box);
    backup.init(frame, box);
    cv2.destroyAllWindows();
    

#run video frame by frame
def framebyframe(file_path):
    
    #A = Average of the BGR channel for the selected area of interest
    # framecount, framenumber, B, G, R, A
    global fps
    framecount = 0
    framenumber = []
    B = []
    G = []
    R = []
    A = []
    cap = cv2.VideoCapture(file_path)
    print(filePath)
    if (cap.isOpened()== False):
        print("Error")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, startframe)
    totalframenumber = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    
    print(totalframenumber)
    #increase framecount by 1 after first frame, glue tracker to roi on first frame
    if (cap.isOpened() == True):
        ret, frame = cap.read()
        if ret == True:
            #increase framecount with 1 after each frame
            framecount = framecount+1
            framenumber.append(framecount)
            roi = cv2.selectROI(frame)
            
        
            frame_cropped = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0] + roi[2]), :]
            
            
            tracker.init(frame, roi);
            backup.init(frame, roi);
            
            channelB, channelG,channelR = cv2.split(frame_cropped);
            B.append(np.mean(channelB))
            G.append(np.mean(channelG))
            R.append(np.mean(channelR))
            A.append((B[-1] + G[-1] + R[-1])/3)
    
    #while video is running for each frame update the tracker based on the area of interest
    while (cap.isOpened()):
        
        ret, frame = cap.read()
        if ret == True:
            ret, roi = tracker.update(frame);
        else:
            print(framenumber[-1])

        if ret == True:
            framecount = framecount+1
            framenumber.append(framecount)

            #get center of roi and use it create an inner frame with size 50x50 
            roi = list(roi)
            if roi[1]<0:
               roi[1] = 1
            frame_cropped = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0] + roi[2]), :]
            c1 = int(roi[3]/2)
            c2 = int(roi[2]/2)
            
            # adjust inner frame size 
            if c1 < ROIsize:
                c1 = ROIsize
                
            if c2 < ROIsize:
                c2 = ROIsize
                
            frame_inner = frame_cropped[c1 - ROIsize:c1+ROIsize, c2 - ROIsize:c2+ROIsize, :]
            
            # take channel averages for the inner frame
         
            
            channelB,channelG,channelR = cv2.split(frame_inner)

            B.append(np.mean(channelB))
            G.append(np.mean(channelG))
            R.append(np.mean(channelR))
            
            
            A.append((B[-1] + G[-1] + R[-1])/3)
            
            frame_cropped = cv2.circle(frame_cropped,(c2,c1), 10, (0,0,255), -1)

            
            frame_cropped = cv2.putText(frame_cropped, "Frame number: " + str(framecount), org, font, fontScale, 
                    font_color, thickness, cv2.LINE_AA);

            cv2.imshow('Frame', frame_cropped)
            cv2.imshow('Inner frame', frame_inner)
            
            if cv2.waitKey(1) & 0xFF == ord('Q'):
                break
            
            if stopframe > 0:
                if framenumber[-1] >= stopframe:
                    break
        # else:
        #     print('Sucess')
        #     break
    cap.release()
    cv2.destroyAllWindows()
    return A, framenumber

#NEW plot digital

def convertBinary(Af):
    arraysbinary = []
    for index, value in enumerate(Af):
        if value < amp_threshold:
            arraysbinary.append(1)
        else:
            arraysbinary.append(0)
    return arraysbinary

def detectBlinks(framenumber, AfBin):
        # blinkCounts = 0
        blinkStartFrame = []
        blinkStopFrame = []
        for n in range (0, len(AfBin)-1):
            if AfBin[n] == 0:
                if AfBin[n+1] ==1:
                    # blinkCounts = blinkCounts + 1
                    blinkStartFrame.append(framenumber[n+1])
            if AfBin[n] ==1:
                if AfBin[n+1] == 0:
                    blinkStopFrame.append(framenumber[n])
                
        return blinkStartFrame, blinkStopFrame
    
def blinkMerge(blinkStartFrame, blinkStopFrame):
    i = 0
    while i <len(blinkStartFrame)-1:
    # for i in range(0, len(blinkStartFrame)-1):
        if blinkStartFrame[i+1] - blinkStopFrame[i] < frame_threshold:
            print(blinkStartFrame[i+1])
            print(blinkStopFrame[i])
            blinkStopFrame[i] = blinkStopFrame[i+1]
            del blinkStartFrame[i+1]
            del blinkStopFrame[i+1]
        i = i+1
    return blinkStartFrame, blinkStopFrame

def createBinary(framenumber, blinkStartFrameMerged, blinkEndFrameMerged):
    binarySignal = []
    print(len(framenumber))

    for i in range(0, len(framenumber)):
        flag = 0
        for g in range(0, len(blinkStartFrameMerged)):
            if i >= blinkStartFrameMerged[g] and i<= blinkEndFrameMerged[g]:
                binarySignal.append(1)
                flag = 1
        if flag == 0:
            binarySignal.append(0)
        
    return binarySignal
            
def estimateDuration(blinkStartFrame, blinkStopFrame):
    blinkDuration = []

    for i in range(len(blinkStartFrame)):
        frameCount = blinkStopFrame[i]-blinkStartFrame[i]
        blinkDuration.append(frameCount / fps)

    return blinkDuration

def createGTbin(framenumber, GTstartblink, GTendblink):
    
    GTbin = []
    
    for i in framenumber:
        flag = 0
        for j in range(0, len(GTstartblink)):
            if i >= GTstartblink[j] and i <= GTendblink[j]:
                GTbin.append(1)
                flag = 1
        if flag==0:
            GTbin.append(0)
            
    return GTbin
#OLD PLOT DIGITAL

# def plotdigital(Af):
#     b = 0
#     arraysbinary = []
#     blinkcheck = False
#     for i in Af:
#         print(i)
#         # while blinkcheck == True:
#         #         if b == i + 20:
#         #             b = 0 and blinkcheck == False
                    
#         if i < -15 and blinkcheck == False:
#             arraysbinary.append(1)
#             blinkcheck = True
#             print(i)
#         else:
#             arraysbinary.append(0)
#     return arraysbinary


def save_data(filePath):
        stefbinary = plotdigital(Af)
    
        print(filePath)
        fileName = filePath.split("/")[-1]
        outputName = fileName.split(".")[0] + '.csv'
        
        with open(outputName, 'w', newline='') as file:
            
            writer = csv.writer(file)
            
            writer.writerow(["Frame number", "Event"])
            
            for i in range(0, len(framenumber)):
                            
                writer.writerow([ framenumber[i], stefbinary[i]])
                

def substractBackground(A):
    Abg = np.convolve(A, np.ones(BGRwinsize), 'valid') / BGRwinsize
    Abg = np.pad(Abg, (BGRwinoffset, BGRwinoffset+1), 'constant', constant_values = (0,0))
    Af = A - Abg # originally 29; -30
    # Aarray = np.asarray(A)
    # arrays = list(map(operator.sub, Abg, Aarray[BGRwinoffset:-1-BGRwinoffset]))
    return Af, Abg

def compareResults(emily, blinkStartFrame):
    #import Emily's annotation and our
    flag = False
    truePositives = []
    falsePositives = []
    startblink = emily['Eye closing Frame'].tolist()
    endblink =  emily['Eye Fully Open'].tolist()
    falseNegatives = startblink.copy()

    
    # for i in range(0, len(startblink)):
    #     for i in range(0, len(endblink)):
    for index, value in enumerate(blinkStartFrame):
        flag = False
        for i in range(0, len(startblink)):
            
            if value >= startblink[i] and value <= endblink[i]:
                flag = True
                truePositives.append(value)
                falseNegatives.remove(startblink[i])
                
        if flag == False:
            falsePositives.append(value)
                    
    print("True positives: ", truePositives)
    print("False positives:", falsePositives)
    print("False negatives:", falseNegatives)
    return truePositives, falsePositives, falseNegatives, startblink, endblink

def saveEvaluation(truePositives, falsePositives, falseNegatives):
    
        fileName = filePath.split("/")[-1]
        outputName = fileName.split(".")[0] + '_Evaluation.csv'
        
        with open(outputName, 'w', newline='') as file:
            
            writer = csv.writer(file)
            
            writer.writerow(["True positives", "False positives", "False negatives"])
            maxAll = max(len(truePositives), len(falsePositives), len(falseNegatives))
            for i in range(0, maxAll):
                bList = []
                if i < len(truePositives):
                    bList.append(truePositives[i])
                else:
                    bList.append(0)
                if i< len(falsePositives):
                    bList.append(falsePositives[i])
                else: 
                    bList.append(0)
                if i< len(falseNegatives):
                    bList.append(falseNegatives[i])
                else:
                    bList.append(0)
                writer.writerow( bList)
    

if __name__ == "__main__":

    
    filePath = openvideofile()
    Av, framenumber = framebyframe(filePath) 
    Avfiltered, Abg = substractBackground(Av)
    # save_data(filePath)
    AvfilteredBinary = convertBinary(Avfiltered)
    bStartFrame, bEndFrame = detectBlinks(framenumber, AvfilteredBinary)
    bStartFrameMerged, bEndFrameMerged = blinkMerge(bStartFrame,bEndFrame)
    blinkMergeBinary = createBinary(framenumber, bStartFrameMerged, bEndFrameMerged)
    print(bStartFrameMerged, bEndFrameMerged)
    # countevents(arraysbinary, framenumber)
    Tp, Fp, Fn, GTstartblink, GTendblink = compareResults(emily, bStartFrameMerged)
    GTbinary = createGTbin(framenumber, GTstartblink, GTendblink)
    saveEvaluation(Tp, Fp, Fn)
    blinkDur = estimateDuration(bStartFrameMerged, bEndFrameMerged)
    
    xmin = 100
    xmax = 700
    ymin = 0
    ymax = 5
    
    plt.figure()
    plt.plot(framenumber, Av, 'k')
    plt.xlim([xmin, xmax])

    plt.figure()
    plt.plot(framenumber, Abg)
    plt.xlim([xmin, xmax])

    plt.figure()
    plt.plot(framenumber, Avfiltered)
    plt.xlim([xmin, xmax])
    
    plt.figure()
    plt.plot(framenumber, (AvfilteredBinary))
    plt.xlim([xmin, xmax])
    plt.ylim([ymin,ymax])
    
    plt.figure()
    plt.plot(framenumber, blinkMergeBinary)
    plt.xlim([xmin, xmax])  
    plt.ylim([ymin,ymax])
    
    plt.figure()
    plt.plot(framenumber, GTbinary)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin,ymax])
        
    # #SAVING DATA AND PLOTTING           
            
    # stefbinary = plotdigital(Af)
    # plt.figure()
    # plt.plot(framenumber, Af, 'y')
    # plt.plot(framenumber, A, 'b')
    # # plt.plot(framenumber, Abg, 'g')
    # plt.plot(framenumber, stefbinary, 'r')
    # save_data(filePath)
    # s, m, n= countevents(stefbinary, framenumber)
    # plt.show()