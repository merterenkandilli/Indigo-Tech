import os
import cv2
import sys
while True:
    try:
        new_width=int(input('Enter new width:'))
        new_height=int(input('Enter new height:'))
        break
    except:
        print("Incorrect dimensions please enter an integer")
        while True:
            cont = input('Would you like to try again?(Y/N):')
            if cont=='N':
                sys.exit()
            else:
                break
path_to_folder=os.path.join(os.getcwd(),'images')
path_to_save=os.path.join(os.getcwd(),'changed_images')
'''
there were 2 folders test and train in images so I used two for loops one for folders one for images
'''
for folders in os.listdir(path_to_folder): # list all folders in folder of interest
    path_to_images=os.path.join(path_to_folder,folders)
    for pictures in os.listdir(path_to_images): # list all images in folder
        if not pictures.endswith('.xml'):
        # check whether it ends with xml or not, it can be changed to if pictures.endswith('.jpg') if exist multiple items in folder
            x=cv2.imread(os.path.join(path_to_images,pictures))
            y=cv2.resize(x,(int(new_width),int(new_height)))
            cv2.imwrite(os.path.join(path_to_save,pictures),y)

print("Succesfully converted to:",new_width,"x",new_height)



