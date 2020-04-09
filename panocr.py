#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:14:52 2020

@author: shirishgupta
"""


import numpy as np
import sys, os
from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request
import io
import cv2

import pytesseract
import re
import matplotlib.pyplot as plt
import math
from pydantic import BaseModel



def blur_and_threshold(gray):
    gray = cv2.GaussianBlur(gray,(3,3),2)
    threshold=cv2.adaptiveThreshold(gray,255,1,1,11,2)
    
    return threshold


def biggest_contour(contours,min_area):
    biggest = None
    max_area = 0
    biggest_n=0
    approx_contour=None
    for n,i in enumerate(contours):
            area = cv2.contourArea(i)
         
            
            if area > min_area/4:
                    peri = cv2.arcLength(i,True)
                    approx = cv2.approxPolyDP(i,0.02*peri,True)
                    if area > max_area and len(approx)==4:
                            biggest = approx
                            max_area = area
                            biggest_n=n
                            approx_contour=approx
                            
                            
                           
    return biggest_n,approx_contour
                            
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    pts=pts.reshape(4,2)
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def select_contour(contours):
    temp=[]
    
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>500:
            temp.append(cnt)
            
    return np.array(temp)
    
def transformation(frame):
    frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_size=gray.size

    threshold=blur_and_threshold(gray)
    edges = cv2.Canny(threshold,50,150,apertureSize = 3)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    biggest_n,approx_contour = biggest_contour(contours,image_size)

    simplified_contours = []
    
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        simplified_contours.append(cv2.approxPolyDP(hull,
                                  0.001*cv2.arcLength(hull,True),True))
        
        
    
    simplified_contours = np.array(simplified_contours)
    
    biggest_n,approx_contour = biggest_contour(simplified_contours,image_size)
        
    frame = cv2.drawContours(frame, simplified_contours ,biggest_n, (0,255,0), 1)
    
 
    dst = 0
    if approx_contour is not None and len(approx_contour)==4:
            
        approx_contour=np.float32(approx_contour)
        dst=four_point_transform(frame,approx_contour)
        #denoise=cv2.fastNlMeansDenoising(dst,dst,50)
    
        
    while True:
        cv2.imshow('frame',dst)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break


    cv2.destroyAllWindows()
    
    return dst


def pan_result(img):
    height, width, channels = img.shape
    croppedImage = img[0:height, 0:int(width/2)] #this line crops
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
    text = pytesseract.image_to_string(croppedImage)
    print(text)
    
    name = None
    fname = None
    dob = None
    pan = None
    nameline = []
    dobline = []
    panline = []
    text0 = []
    text1 = []
    text2 = []
    lineno = 0
    lines = text.split('\n')
    for lin in lines:
        s = lin.strip()
        s = s.rstrip()
        s = s.lstrip()
        s = re.sub(r"/", 'slash', s)
        s = re.sub(r"[^a-zA-Z0-9]+", ' ', s)
        s = re.sub(r"slash", '/', s)
        text1.append(s)
    
    text1 = list(filter(None, text1)) 
    
    for wordline in text1:
        xx = wordline.split('\n')
        if ([w for w in xx if re.search('(INCOME|TAX|INCOMETAX|GOW|GOVT|GOVERNMENT|OVERNMENT|VERNMENT|DEPARTMENT|EPARTMENT|PARTMENT|ARTMENT|INDIA|NDIA)$', w)]):
            text1 = list(text1)
            lineno = text1.index(wordline)
            break
    if(lineno == 0):
      text0 = text1
    else:
      text0 = text1[lineno+1:]
    
    try:
        for x in text0:
            for y in x.split():
                nameline.append(x)
                break
    except:
        pass
    
    try:
        name = nameline[0]
        fname = nameline[1]
        dob = nameline[2]
        pan = nameline[4]
    except:
        pass
    
    # Making tuples of data
    data = {}
    data['Name'] = name
    data['Father Name'] = fname
    data['Date of Birth'] = dob
    
    
    
    words = text.split()
    for i in range(len(words)):
      if(words[i].isalnum()):
        if(len(words[i])==10):
          if(words[i][5].isdigit() & words[i][6].isdigit() & words[i][7].isdigit() & words[i][8].isdigit()):
            #print(words[i])
            data['PAN'] = words[i]
   
    #print(data)
    
    return(data) 


#img = cv2.imread("/Users/shirishgupta/Documents/Loan2Grow/FacialRecognition/Pancard/Pancard Photos/Pancard4.jpg")

#pan_result(img)

app = FastAPI()

class ImageType(BaseModel):
    url: str


@app.post("/predict/")    
def prediction(request: Request, 
	file: bytes = File(...)):

	if request.method == "POST":
		image_stream = io.BytesIO(file)
		image_stream.seek(0)
		file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
		frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
		label = pan_result(frame)
		return label
	return "No post request found"