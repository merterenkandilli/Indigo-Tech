import os
import pandas as pd
import cv2


path_to_images=str(input('Enter path to images:  '))
path_to_txt=str(input('Enter path to txt:  '))
type = str(input('Enter type of csv (train/test):  '))
columns= ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
rows=[]
path_to_images=os.path.join(os.getcwd(),path_to_images)
path_to_txt=os.path.join(os.getcwd(),path_to_txt)
for txt,images in zip(os.listdir(path_to_txt),os.listdir(path_to_images)):
    path_to_txt1 = os.path.join(path_to_txt, txt)
    path_to_image=os.path.join(path_to_images,images)
    img=cv2.imread(path_to_image)
    height, width = img.shape[:2]
    '''open() reads the txt file .readlines() read the file line by line'''
    with open(path_to_txt1) as f:
        lines = f.readlines()
    for coordinates in lines:
        coord=str(coordinates)
        '''.split() break the txt file and stores as array ' ' is used to split by spaces 6 means it is divided in 6 piece.'''
        coord=coord.split(' ',6)
        xmin,ymin,w,h,_,_= coord
        xmax=int(xmin)+int(int(w)/2)
        ymax=int(ymin)+int(int(h)/2)
        xmin=int(xmin)-int(int(w)/2)
        ymin=int(ymin)-int(int(h)/2)
        # if added coordinates exceeding size of image it does not take that coordinate
        # The dataset I used was head annotated so I needed to check.
        if int(xmin)>int(width) or int(xmax)>int(width) or int(ymin)>int(height) or int(ymax)>int(height):
            continue
        else:
            row = (images,
                   width,  # width
                   height,  # hight
                   'person',  # class
                   xmin,  # xmin
                   ymin,  # ymin
                   xmax,  # xmax
                   ymax)  # ymax
            rows.append(row)
data_frame = pd.DataFrame(rows, columns=columns) #converts array to table.
data_frame.to_csv('data/{}_labels.csv'.format(type), index=None)
print('Successfully converted from txt to csv.')