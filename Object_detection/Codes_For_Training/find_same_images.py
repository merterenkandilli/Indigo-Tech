import os
import cv2
import sys
while True:
        images_path=input('Enter images folder:')
        save_path=input('Enter save folder:')
        if not os.path.exists(images_path) or not os.path.exists(save_path):
            print("One or both of the folders does not exist!!")
            while True:
                cont = input('Would you like to try again?(Y/N):')
                if cont=='N':
                    sys.exit()
                else:
                    break
        else:
            break
path_to_folder=os.path.join(os.getcwd(),images_path)
path_to_save=os.path.join(os.getcwd(),save_path)

for picture in os.listdir(path_to_folder):
    if picture.endswith('.jpg'):
        picture_resized1 = cv2.imread(os.path.join(path_to_folder, picture))
        picture_resized1 = cv2.resize(picture_resized1, (800, 800))
        for picture2 in os.listdir(path_to_folder):
            if picture.endswith('.jpg'):
                picture_resized2=cv2.imread(os.path.join(path_to_folder, picture2))
                picture_resized2=cv2.resize(picture_resized2,(800,800))
                if picture!=picture2:
                    diff=cv2.absdiff(picture_resized1,picture_resized2) # it looks difference in two images
                    x=len([1 for i in diff[0] if all(i)==0])
                    if x==len(diff[0]):    # if two images are same difference returns 0 array
                        cv2.imwrite(os.path.join(path_to_save, picture), picture_resized1)
                        cv2.imwrite(os.path.join(path_to_save, picture2), picture_resized2)
