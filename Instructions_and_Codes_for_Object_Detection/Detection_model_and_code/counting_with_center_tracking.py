import os
from collections import OrderedDict
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np
from math import sqrt
import scipy.spatial.distance

model_name='model_person_top_19k'
#first we take the path of the necessary folders.
model_direction=os.path.join(os.getcwd(),os.path.join(model_name,'saved_model'))
path_to_config=os.path.join(os.getcwd(),os.path.join(model_name,'pipeline.config'))
path_to_labels=os.path.join(os.getcwd(),'label_map.pbtxt')
path_to_checkpoint=os.path.join(os.getcwd(),os.path.join(model_name,'checkpoint/'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging


tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

configurations=config_util.get_configs_from_pipeline_file(path_to_config) #reads entire config file.
model_configuration=configurations['model']  #reads the code in model{ .. }
detection_model = model_builder.build(model_config=model_configuration, is_training=False)
#as far as I understand model_builder.build constructs an flow chart.

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(path_to_checkpoint, 'ckpt-0')).expect_partial()
counts=0
@tf.function
def detection_process(image):
    """
    image and shapes returns to array. My assumption is that .preprocess returns to orijinal image and detection locations.
    .predict compute the accuracy of detection.
    .postprocess returns to accuracy and locations.
    """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes) # it is used to compute class probabilities
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections
def check_new_object(old_centers,centers,count,num_detect,check_number):
    euclidean_dist=scipy.spatial.distance.cdist(old_centers, centers)
    for i in range(num_detect):
        if min(euclidean_dist[i][0:num_detect]) <= 200:
            continue
        else:
            if min(euclidean_dist[i][0:num_detect]) not in check_number:
                check_number[i] = min(euclidean_dist[i][0:num_detect])
                count = count + 1
    return count
def check_number_of_frame(frame_number,centers):
    frame_number=frame_number+1
    if frame_number==10:
        centers = np.zeros((10, 2), dtype="int")
        frame_number=0
    return frame_number,centers


def centroid_calculation(detections,height,width,centers,count,old_centers,check_number):
    num_detect=0
    for i in range(10):
        if detections['detection_scores'][0][i].numpy()>0.5:
            ymin = int(detections['detection_boxes'][0][i][0].numpy() * height)
            xmin = int(detections['detection_boxes'][0][i][1].numpy() * width)
            ymax = int(detections['detection_boxes'][0][i][2].numpy() * height)
            xmax = int(detections['detection_boxes'][0][i][3].numpy() * width)
            if 190<ymin<210:
                center_x=(xmin+xmax)/2
                center_y=(ymax+ymin)/2
                centers[num_detect]=(center_x,center_y)
                num_detect=num_detect+1

    count=check_new_object(old_centers,centers,count,num_detect,check_number)
    for i in range(10):
        old_centers[i] = centers[i]
    return count,centers

labels = label_map_util.create_category_index_from_labelmap(path_to_labels,use_display_name=True)

font = cv2.FONT_HERSHEY_SIMPLEX
org = (120, 50)
thickness = 1
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255)]
centers=np.zeros((10,2),dtype="int")
old_centers=np.zeros((10,2),dtype="int")
check_number = np.zeros(10)
frame_number=0
path_to_video="stock-footage-top-down-view-of-london-city-workers-on-their-commute-from-the-office-at-the-end-of-the-working-day.webm"
cap = cv2.VideoCapture(path_to_video) # used to capture video
while True:
    ret, image_np = cap.read() # Reads captured video returns to true and video
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32) #conversion to tensor
    #Tensorflow reads the images as float32 type arrays so we convert the input with above code
    # np.expand_dims convert the image to numbers in array as RGB code.
    detections = detection_process(input_tensor)
    location_of_label = 1 # its about the location of classes on pictures when I changed to another number it gives N/A
    image_np_with_detections = image_np.copy()

    #opensource code to draw boxes, it takes boxes,classes,scores and draw the boxes if accuracy is bigger than min_score_thresh.
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + location_of_label).astype(int),
          detections['detection_scores'][0].numpy(),
          labels,
          use_normalized_coordinates=True,
          line_thickness=2,
          max_boxes_to_draw=200,
          min_score_thresh=.50,
          agnostic_mode=False)
    height, width = image_np_with_detections.shape[:2]   #(360,640)
    counts,centers=centroid_calculation(detections, height, width,centers,counts,old_centers,check_number)
    frame_number, centers=check_number_of_frame(frame_number, centers)
    cv2.putText(image_np_with_detections, str(counts), org, font,1, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(image_np_with_detections, 'Count:', (20, 50), font,1, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.rectangle(image_np_with_detections,(0,190),(595,240),(0,0,255),1)

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (int(width), int(height)))) #cv2 code to display video.

    if cv2.waitKey(1) & 0xFF == ord('q'):  # number inside waitkey indicates the number of miliseconds that video will be displayed
        break
cap.release()
cv2.destroyAllWindows()