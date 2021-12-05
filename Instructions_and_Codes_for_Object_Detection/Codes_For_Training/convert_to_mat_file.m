%it takes csv files coming from my head_coordinates.py and converts them
%.mat file to generate density map. I used paper's density generation so I
%convert the table to 1x1 cell which contains location and number of people

annotations = readtable('crowd-tourists-260nw-379057_labels.csv','PreserveVariableNames',true);
number_of_people=size(annotations,1);
annotations1=table2array(annotations);
image_info=cell(1);
image_info{1}=struct;
image_info{1}.location=annotations1;
image_info{1}.number=number_of_people;