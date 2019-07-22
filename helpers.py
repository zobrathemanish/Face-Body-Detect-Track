#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:51:33 2017

@author: kyleguan
"""
import numpy as np
import cv2
import os
import time

import Age_Gender.eval as ager

import faceConfidence.faceConfidence as faceconf

from Person import insertPerson

global all_folders
all_folders = []

shape_detector = 'Age_Gender/shape_predictor_68_face_landmarks.dat'
model_path = './models'

id_list = []

line = [(0, 650), (1500, 650)]


class Box:


    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()


# Return true if line segments AB and CD intersect


def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);



def box_iou2(a, b):
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''
    
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])
  
    return float(s_intsec)/(s_a + s_b -s_intsec)



def convert_to_pixel(box_yolo, img, crop_range):
    '''
    Helper function to convert (scaled) coordinates of a bounding box 
    to pixel coordinates. 
    
    Example (0.89361443264143803, 0.4880486045564924, 0.23544462956491041, 
    0.36866588651069609)
    
    crop_range: specifies the part of image to be cropped
    '''
    
    box = box_yolo
    imgcv = img
    [xmin, xmax] = crop_range[0]
    [ymin, ymax] = crop_range[1]
    h, w, _ = imgcv.shape
    
    # Calculate left, top, width, and height of the bounding box
    left = int((box.x - box.w/2.)*(xmax - xmin) + xmin)
    top = int((box.y - box.h/2.)*(ymax - ymin) + ymin)
    
    width = int(box.w*(xmax - xmin))
    height = int(box.h*(ymax - ymin))
    
    # Deal with corner cases
    if left  < 0    :  left = 0
    if top   < 0    :   top = 0
    
    # Return the coordinates (in the unit of the pixels)
  
    box_pixel = np.array([left, top, width, height])
    return box_pixel



def convert_to_cv2bbox(bbox, img_dim = (1280, 720)):
    '''
    Helper fucntion for converting bbox to bbox_cv2
    bbox = [left, top, width, height]
    bbox_cv2 = [left, top, right, bottom]
    img_dim: dimension of the image, img_dim[0]<-> x
    img_dim[1]<-> y
    '''
    left = np.maximum(0, bbox[0])
    top = np.maximum(0, bbox[1])
    right = np.minimum(img_dim[0], bbox[0] + bbox[2])
    bottom = np.minimum(img_dim[1], bbox[1] + bbox[3])
    
    return (left, top, right, bottom)
    
    
def draw_box_label(id,img, bbox_cv2, box_color=(0, 255, 255), show_label=True):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]
    
    # Draw the bounding box
    # cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)

    
    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        # cv2.rectangle(img, (left-2, top-45), (right+2, top), box_color, -1, 1)
        # print([left, bottom])

        # cv2.line(img, line[0], line[1], (0, 255, 255), 5)
        
        # Output the labels that show the x and y coordinates of the bounding box center.
        text_x= 'id='+str(id)
        # cv2.putText(img,text_x,(left,top-25), font, font_size, font_color, 1, cv2.LINE_AA)
        text_y= 'y='+str((top+bottom)/2)
        # print(text_y)
        centroid = (top+bottom)/2

        if((bottom > 1042) and (bottom) < 1050):

            xima = left
            yima = top
            widther = right - left
            heighter = bottom - top 

            drawer = img.copy()
            crop_img = drawer[yima:yima+heighter, xima:xima+widther]

            folder = os.path.exists('DB/' + str(id))

            if(folder == False):

                os.makedirs('DB/' + str(id))

            image_path = 'DB/' + str(id) + '/' + str(time.time()) + ".jpg"

            cv2.imwrite(image_path,crop_img)

            # r = faceconf.confidence_score(image_path)

            # with open('DB/' + str(id) + '/'  + str(id) + ".txt", "a") as myfile:
                
            #     myfile.write(str(r) + "\n")

                

                # if str(id) not in all_folders:

                #     all_folders.append(str(id))

                # print(all_folders)


                # gimber.append(str(id))


            # time.sleep(500)

            # while(False):

            try:

                aligned_image, image, rect_nums, XY = ager.load_image(image_path, shape_detector)

                ages, genders = ager.eval(aligned_image, model_path)

                # print(str(id), ages, genders)

                age = int(ages)
                gen = int(genders)

                if(gen == 0):
                    real_gender = "Female"
                else:
                    real_gender = "Male"


                with open('DB/' + str(id) + '/'  + str(id) + ".txt", "a") as myfile:
                    # print("Writing the intended text")
                    myfile.write(str(id) + ' ' + str(age) + ' ' + str(real_gender) +  "\n")

                print(str(id) , "Images are good !!")
                    
                insertPerson(str(age),real_gender,str(id)+"/"+str(id)+".jpg")

            except Exception:

                with open('DB/' + str(id) + '/'  + str(id) + ".txt", "a") as myfile:
                    # print("Writing the intended text")

                    myfile.write(str(id) + " Internal Server Error" +  "\n")
                    # myfile.write(str(id) + " Image is too low in resolution" +  "\n")
                
                print(str(id) , "The images are too low for resolution !!")


                insertPerson("15","m","manish/bottle")

            if text_x not in id_list:
                id_list.append(text_x)

        # cv2.putText(img,text_y,(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)

        # print(id_list)

    return img
