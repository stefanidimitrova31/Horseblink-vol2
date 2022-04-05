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

#global variables
ROIsize = 100 # area of interest size
BGRwinsize = 60 # framerate per second is 60
BGRwinoffset = int(BGRwinsize/2)-1
stopframe = int(input('Chose frame to stop at:'))

# if the number of frames the user has set is less than 30 make it equal to 30, the program requires at least 30 frames to
# perform adequate analysis for the background
if stopframe < BGRwinsize:
    stopframe = BGRwinsize
    print('Error, stopframe is changed to:', stopframe)

#function that allows us to select file
def openvideofile():
#the root window into which all other widgets go. It is an instance of the class Tk
    root = tk.Tk()
#Removes the window from the screen (without destroying it)
    root.withdraw()

    file_path = filedialog.askopenfilename()
    return file_path

#font specification

font = cv2.FONT_HERSHEY_SIMPLEX;
org = (50, 50);
fontScale = 1;
font_color = (255, 255, 0);
thickness = 2;


#create tracker

tracker = cv2.TrackerCSRT_create(); 
backup = cv2.TrackerCSRT_create();


#Retrieve first frame where we select area of interest
def initialize_camera(cap):
    _, frame = cap.read()
    return frame

    box = cv2.selectROI(frame, False);
    tracker.init(frame, box);
    backup.init(frame, box);
    cv2.destroyAllWindows();
    

#run video frame by frame
def framebyframe(filePath):
    
    #A = Average of the BGR channel for the selected area of interest
    global framecount, framenumber, B, G, R, A
    framecount = 0
    framenumber = []
    B = []
    G = []
    R = []
    A = []
    cap = cv2.VideoCapture(filePath)
    print(filePath)
    if (cap.isOpened()== False):
        print("Error")
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
        
        ret, roi = tracker.update(frame);

        if ret == True:
            framecount = framecount+1
            framenumber.append(framecount)

            #get center of roi and use it create an inner frame with size 50x50 
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

            channelB, channelG,channelR = cv2.split(frame_inner);
            
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



if __name__ == "__main__":


    filePath = openvideofile()
    framebyframe(filePath) 
    
    
    Abg = np.convolve(A, np.ones(BGRwinsize), 'valid') / BGRwinsize
    Abg = np.pad(Abg, (BGRwinoffset, BGRwinoffset+1), 'constant', constant_values = (0,0))
    Af = A - Abg # originally 29; -30
    Aarray = np.asarray(A)
    arrays = list(map(operator.sub, Abg, Aarray[BGRwinoffset:-1-BGRwinoffset]))
    
    plt.figure()   
    plt.plot(framenumber, B, color = 'blue')
    plt.plot(framenumber, G, color = 'green')
    plt.plot(framenumber, R, color = 'red')
    plt.plot(framenumber, A, color = 'black')
    
    def save_data(filePath):
        # framenumbers = np.array(framenumber)
    
        print(filePath)
        fileName = filePath.split("/")[-1]
        outputName = fileName.split(".")[0] + '.csv'
        
        with open(outputName, 'w', newline='') as file:
            
            writer = csv.writer(file)
            
            writer.writerow(["Frame number", "Event"])
            
            for i in range(0, len(framenumber)):
                            
                writer.writerow([ framenumber[i], stefbinary[i]])
                        
    #plot digital
    def plotdigital(Af):
        arraysbinary = []
        for i in Af:
            if i <  -5:
                arraysbinary.append(1)
            else:
                arraysbinary.append(0)
        return arraysbinary     
    
    def countevents(arraysbinary, framenumber):
        count = 0
        blinkStartFrame = []
        blinkStopFrame = []
        for n in range (0, len(arraysbinary)-1):
            if arraysbinary[n] == 0:
                if arraysbinary[n+1] ==1:
                    count = count + 1
                    blinkStartFrame.append(framenumber[n+1])
            if arraysbinary[n] ==1:
                if arraysbinary[n+1] == 0:
                    blinkStopFrame.append(framenumber[n])
                
        return count, blinkStartFrame, blinkStopFrame
    
        
    #SAVING DATA AND PLOTTING           
            
    stefbinary = plotdigital(Af)
    plt.figure()
    plt.plot(framenumber, Af, 'y')
    plt.plot(framenumber, A, 'b')
    plt.plot(framenumber, Abg, 'g')
    plt.plot(framenumber, stefbinary, 'r')
    save_data(filePath)
    s, m, n= countevents(stefbinary, framenumber)
    plt.show()