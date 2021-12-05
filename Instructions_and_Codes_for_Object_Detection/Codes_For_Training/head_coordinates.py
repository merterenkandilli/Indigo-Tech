import os
import sys
import cv2
import easygui
import pandas as pd

columns=['x','y']
quit=False
while True:
    '''
    easygui is used to choose the image to be annotated.
    '''
    while True:
        image_path = easygui.fileopenbox()
        '''
        if image_path return to none which means user cancelled the process and annotation process ended.
        else image is chosen with incorrect extension, process starts over and code print reason of error.
        if image is chosen correctly than loop is broken and process starts. 
        '''
        if image_path == None:
            sys.exit()
        if image_path.endswith('.jpg'):
            break
        else:
            print('Image extension should be .jpg')

    coordinates = []
    '''
    img need to be decleared globally since we might delete the last chosen coordinate 
    when it is decleared locally it was raising error and image stayed same
    '''
    def mouse_click_check(event, x, y, flags, params):
        global img
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append((x, y))
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.imshow('image', img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            img = cv2.imread(image_path)
            coordinates.pop()
            for coord in coordinates:
                x, y = coord
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.imshow('image', img)
    global img
    while True:
        img = cv2.imread(image_path)
        cv2.imshow('image', img)
        '''
        setMousecallback function is an opencv code which returns which button of mouse is clicked and coordinate of action
        '''
        cv2.setMouseCallback('image', mouse_click_check)

        if cv2.waitKey(0) & 0xFF ==ord('s'):
            '''
            pandas dataframe is used in conversion to csv file it takes two inputs columns value and 
            rows value and generate rows and columns
            '''
            rows_column = pd.DataFrame(coordinates, columns=columns)
            name=os.path.splitext(image_path)[0]
            rows_column.to_csv('{}_labels.csv'.format(name), index=None)
        if cv2.waitKey(0) & 0xFF == ord('n'):
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            quit=True
            break
    if quit==True:
        sys.exit()
    if image_path == None:
        sys.exit()

