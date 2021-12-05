import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
# We need filename, width, height,class,xmin, ymin,xmax,ymax of xml files that we created using label image
# So construct columuns array with those.
columns= ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
# Also we need to construct a row array for values corresponding to these columns.
rows=[]
def folder_extraction():
    for folders in os.listdir('images'): # os.listdir finds all subfolders in a folder
        xmlfiles=os.path.join(os.getcwd(),'images/{}'.format(folders))  # os.path.join joins two path in my code it takes directory of
                                                                        # .py file by os.getcwd
        xml_dataframe=csv_conversion(xmlfiles)
        xml_dataframe.to_csv('csv_owncode/{}_label.csv'.format(folders), index=None) #this syntax converts to csv file
        #Need to create a csv_owncode directory it does not create by default.
        print('Successfully converted xml to csv.')

def csv_conversion(xmlfiles):
    for xmlfile in os.listdir(xmlfiles):
        if xmlfile.endswith('.xml'):
            filename=os.path.join(xmlfiles,xmlfile)
            parsedxml=ET.parse(filename)  #First we parse the xml file then
            roots_of_parsed_xml=parsedxml.getroot() #We use file.getroot() function to reach one level below of an element.
            #with the syntax .find we find the index of root and treat its submodules as array to find searched value.
            #we need to use for loop in case there exist same object more than ones in one picture.
            adress_of_objects=roots_of_parsed_xml.findall('object')   #with this we find adress of objects
            for i in range(len(adress_of_objects)):
                row= (roots_of_parsed_xml.find('filename').text,
                     int(roots_of_parsed_xml.find('size')[0].text), #width
                     int(roots_of_parsed_xml.find('size')[1].text), #hight
                     adress_of_objects[i][0].text, # class
                     int(adress_of_objects[i][4][0].text), #xmin
                     int(adress_of_objects[i][4][1].text), #ymin
                     int(adress_of_objects[i][4][2].text), #xmax
                     int(adress_of_objects[i][4][3].text)) #ymax
                rows.append(row)
        else:
            continue
    data_frame = pd.DataFrame(rows, columns=columns)             #using pandas DataFrame syntax to construct columns and rows
    return data_frame
folder_extraction()
