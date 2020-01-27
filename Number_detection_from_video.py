# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:13:01 2020

@author: naveenn
"""
import os
import numpy as np
import cv2
from skimage.transform import resize
from skimage.filters import threshold_local
import imutils

os.chdir(r'C:\Users\Public\Documents\Python Scripts\Number_plate_detection')

filename = './video12.mp4'

cap = cv2.VideoCapture(filename)
# cap = cv2.VideoCapture(0)
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    if ret == True:
        cv2.imshow('window-name',frame)
        cv2.imwrite("./output/frame%d.jpg" % count, frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


#from PIL import Image
#im1 = Image.open(img_path)
#width, height = im1.size
#if(width > 600):
#    im1 = im1.resize((700,600), Image.NEAREST)
#    im1.save(img_path)

img = cv2.imread("./output/frame%d.jpg"%(count-1), as_gray=True)
#img = 255 - img

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.bilateralFilter(img_gray,5,50,50) 
#img_gray = cv2.blur(img_gray,(2,2)) 
WHITE = [255, 255, 255]
img_gray = cv2.copyMakeBorder(img_gray, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=WHITE)

#img_gray = cv2.Sobel(img_gray,cv2.CV_8U,1,0,ksize=3)
img_bw = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY)[1]
#img_bw = cv2.blur(img_bw,(3,3)) 

#cv2.imshow('BW Image', img_bw)
#cv2.waitKey(0)

from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# this gets all the connected regions and groups them together
label_image = measure.label(img_bw)

# print(label_image.shape[0]) #width of car img

# getting the maximum width, height and minimum width and height that a license plate can be
plate_dimensions = (0.03*label_image.shape[0], 0.08*label_image.shape[0], 0.15*label_image.shape[1], 0.3*label_image.shape[1])
plate_dimensions2 = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []
plate_like_objects = []

fig, (ax1) = plt.subplots(1)
ax1.imshow(img_gray, cmap="gray")
flag =0
# regionprops creates a list of properties of all the labelled regions
for region in regionprops(label_image):
    # print(region)
    if region.area < 50:
        #if the region is so small then it's likely not a license plate
        continue
        # the bounding box coordinates
    min_row, min_col, max_row, max_col = region.bbox

    region_height = max_row - min_row
    region_width = max_col - min_col

    # ensuring that the region identified satisfies the condition of a typical license plate
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        flag = 1
        plate_like_objects.append(img_bw[min_row:max_row,
                                  min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col,
                                         max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                       linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
        # let's draw a red rectangle over those regions
if(flag == 1):
    # print(plate_like_objects[0])
    plt.show()

if(flag==0):
    min_height, max_height, min_width, max_width = plate_dimensions2
    plate_objects_cordinates = []
    plate_like_objects = []

    fig, (ax1) = plt.subplots(1)
    ax1.imshow(img_gray, cmap="gray")

    # regionprops creates a list of properties of all the labelled regions
    for region in regionprops(label_image):
        if region.area < 50:
            #if the region is so small then it's likely not a license plate
            continue
            # the bounding box coordinates
        min_row, min_col, max_row, max_col = region.bbox

        region_height = max_row - min_row
        region_width = max_col - min_col

        # ensuring that the region identified satisfies the condition of a typical license plate
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            # print("hello")
            plate_like_objects.append(img_bw[min_row:max_row,
                                      min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col,
                                             max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="green",
                                           linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
            # let's draw a red rectangle over those regions
    # print(plate_like_objects[0])
    plt.show()

#############===== Extracting Number Plate =============################

if (len(plate_objects_cordinates)>1):
    len_plate = len(plate_objects_cordinates)
    for index,cordinates in enumerate(plate_objects_cordinates):
        first = cordinates[0]
        second = cordinates[2]
        third = cordinates[1]
        forth = cordinates[3]
        
#        name = 'plate_'+str(index)+'.jpg'
        plate = img[first:second, third:forth] 
        
        V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 29, offset=15, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)
         
        # resize the license plate region to a canonical size
        plate = imutils.resize(plate, width=350)
        thresh = imutils.resize(thresh, width=350)
                
        labels = measure.label(thresh, neighbors=4, background=0)
        #charCandidates = np.zeros(thresh.shape, dtype="uint8")
        
        license_plate = thresh
        labelled_plate = measure.label(license_plate)
        
        fig, ax1 = plt.subplots(1)
        ax1.imshow(license_plate, cmap="gray")
        # the next two lines is based on the assumptions that the width of
        # a license plate should be between 5% and 15% of the license plate,
        # and height should be between 35% and 60%
        # this will eliminate some
        character_dimensions = (0.40*license_plate.shape[0], 0.70*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
        min_height, max_height, min_width, max_width = character_dimensions
        
        characters = []
        counter=0
        column_list = []
        for regions in regionprops(labelled_plate):
            y0, x0, y1, x1 = regions.bbox
            region_height = y1 - y0
            region_width = x1 - x0
        
            if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
                roi = license_plate[y0:y1, x0:x1]
        
                # draw a red bordered rectangle over the character.
                rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                               linewidth=2, fill=False)
                ax1.add_patch(rect_border)
        
                # resize the characters to 20X20 and then append each character into the characters list
                resized_char = resize(roi, (30, 30))
                characters.append(resized_char)
        
                # this is just to keep track of the arrangement of the characters
                column_list.append(x0)
        
        if(len(characters) > 0):
            cv2.imshow('img', plate)
            cv2.waitKey(0)
            cv2.imwrite(os.path.join('Number_plates','number_plate.jpg'), plate)
            
            cv2.imshow('img', thresh)
            cv2.waitKey(0)
            break
                
else:
    first = plate_objects_cordinates[0][0]
    second = plate_objects_cordinates[0][2]
    third = plate_objects_cordinates[0][1]
    forth = plate_objects_cordinates[0][3]
    
    plate = img[first:second, third:forth]    
#    cv2.imshow('img', plate)
#    cv2.waitKey(0)    
    cv2.imwrite(os.path.join('Number_plates','number_plate.jpg'), plate)

#######====== Convert to string ========############
    
    V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 29, offset=15, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)
     
    # resize the license plate region to a canonical size
    plate = imutils.resize(plate, width=350)
    thresh = imutils.resize(thresh, width=350)
    
#    cv2.imshow('img', thresh)
#    cv2.waitKey(0)
    
    labels = measure.label(thresh, neighbors=4, background=0)
    #charCandidates = np.zeros(thresh.shape, dtype="uint8")
    
    license_plate = thresh
    labelled_plate = measure.label(license_plate)
    
    fig, ax1 = plt.subplots(1)
    ax1.imshow(license_plate, cmap="gray")
    # the next two lines is based on the assumptions that the width of
    # a license plate should be between 5% and 15% of the license plate,
    # and height should be between 35% and 60%
    # this will eliminate some
    character_dimensions = (0.30*license_plate.shape[0], 0.70*license_plate.shape[0], 0.05*license_plate.shape[1], 0.10*license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions
    
    characters = []
    counter=0
    column_list = []
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0
    
        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]
    
            # draw a red bordered rectangle over the character.
            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                           linewidth=2, fill=False)
            ax1.add_patch(rect_border)
    
            # resize the characters to 20X20 and then append each character into the characters list
            resized_char = resize(roi, (30, 30))
            characters.append(resized_char)
    
            # this is just to keep track of the arrangement of the characters
            column_list.append(x0)
    # print(characters)
    plt.show()


import pickle
print("Loading model")
filename = './finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

print('Model loaded. Predicting characters of number plate')
classification_result = []

for each_character in characters:
    # converts it to a 1D array
    each_character = each_character.reshape(1, -1)
    result = model.predict(each_character)
    classification_result.append(result)

print('Classification result')
print(classification_result)
 
plate_string = ''
for eachPredict in classification_result:
    plate_string += eachPredict[0]

#print('Predicted license plate')
#print(plate_string)

# it's possible the characters are wrongly arranged
# since that's a possibility, the column_list will be
# used to sort the letters in the right order

column_list_copy = column_list[:]
column_list.sort()
rightplate_string = ''
for each in column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print('\nNumber plate')
print(rightplate_string)

def spen(r,h,c,e,s,t):
    return s-(r+h+c+e+t)
    
spen(1510,10000,0,4444,28000+2000,1000)



