# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:58:37 2019

@author: naveenn
"""

from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from skimage.transform import resize
from skimage.filters import threshold_local
import imutils
from skimage import measure
from skimage.measure import regionprops

app = Flask(__name__)

@app.route('/')
def session():
    return render_template('index.html')

@app.route('/upload', methods = ['GET', 'POST'])
def upload_img():
    global file_path
    #if request.method == 'POST':
    f = request.files['image_uploads']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'static/uploads', secure_filename(f.filename))
    f.save(file_path)
    
    return render_template('index.html', image = file_path)

@app.route('/Get_number_plate', methods = ['GET', 'POST'])
def get_number_plate():
    os.chdir(r'C:\Users\Public\Documents\Python Scripts\Number_plate_detection\Flask')
    img_path = file_path

    from PIL import Image
    im1 = Image.open(img_path)
    width, height = im1.size
    if(width > 600):
        im1 = im1.resize((700,600), Image.NEAREST)
        im1.save(img_path)

    img = cv2.imread(img_path)
#---- Image Processing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bilateralFilter(img_gray,5,50,50) 
    WHITE = [255, 255, 255]
    img_gray = cv2.copyMakeBorder(img_gray, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=WHITE)
    img_bw = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY)[1]
    
#---- Getting connected regions    
    label_image = measure.label(img_bw)    
    plate_dimensions = (0.03*label_image.shape[0], 0.08*label_image.shape[0], 0.15*label_image.shape[1], 0.3*label_image.shape[1])
    plate_dimensions2 = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions
#---- Finding Plate
    plate_objects_cordinates = []
    plate_like_objects = []
    
    flag = 0
    for region in regionprops(label_image):
        if region.area < 50:
            continue

        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col

        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            flag = 1
            plate_like_objects.append(img_bw[min_row:max_row, min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col, max_row, max_col))
        
    if(flag==0):
        min_height, max_height, min_width, max_width = plate_dimensions2
        plate_objects_cordinates = []
        plate_like_objects = []
    
        for region in regionprops(label_image):
            if region.area < 50:
                continue
            min_row, min_col, max_row, max_col = region.bbox
            
            region_height = max_row - min_row
            region_width = max_col - min_col

            if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
                plate_like_objects.append(img_bw[min_row:max_row, min_col:max_col])
                plate_objects_cordinates.append((min_row, min_col, max_row, max_col))

#---- Getting cordinates                
    if (len(plate_objects_cordinates)>1):
        for index,cordinates in enumerate(plate_objects_cordinates):
            first = cordinates[0]
            second = cordinates[2]
            third = cordinates[1]
            forth = cordinates[3]
        
            plate = img[first:second, third:forth] 
        
            V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
            T = threshold_local(V, 29, offset=15, method="gaussian")
            thresh = (V > T).astype("uint8") * 255
            thresh = cv2.bitwise_not(thresh)
         
            plate = imutils.resize(plate, width=350)
            thresh = imutils.resize(thresh, width=350)
        
            license_plate = thresh
            labelled_plate = measure.label(license_plate)
        
            character_dimensions = (0.40*license_plate.shape[0], 0.70*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
            min_height, max_height, min_width, max_width = character_dimensions
        
            characters = []
            column_list = []
            for regions in regionprops(labelled_plate):
                y0, x0, y1, x1 = regions.bbox
                region_height = y1 - y0
                region_width = x1 - x0
                
                if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
                    roi = license_plate[y0:y1, x0:x1]
                    
                    resized_char = resize(roi, (30, 30))
                    characters.append(resized_char)
                    
                    column_list.append(x0)
        
            if(len(characters) > 0):
                cv2.imwrite(os.path.join('static','Plate_img/number_plate.jpg'), plate)
                rectangle = cv2.rectangle(img, (third,first), (forth, second), (0,0,255), 2)
                cv2.imwrite(os.path.join('static','Plate_img/car_with_plate.jpg'), rectangle)
                break
                
    else:
        first = plate_objects_cordinates[0][0]
        second = plate_objects_cordinates[0][2]
        third = plate_objects_cordinates[0][1]
        forth = plate_objects_cordinates[0][3]
        
        plate = img[first:second, third:forth]    
        cv2.imwrite(os.path.join('static','Plate_img/number_plate.jpg'), plate)
        rectangle = cv2.rectangle(img, (third,first), (forth, second), (0,0,255), 2)
        cv2.imwrite(os.path.join('static','Plate_img/car_with_plate.jpg'), rectangle)
                
    return render_template('index.html', plate_image = os.path.join('static','Plate_img/number_plate.jpg'))
    
    
if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host= '0.0.0.0', debug = True)
