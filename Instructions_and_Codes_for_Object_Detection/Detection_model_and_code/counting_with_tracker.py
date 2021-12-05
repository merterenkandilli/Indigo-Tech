import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np
import math

model_name='side_rcnn'
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
def draw_rectangle_boxes(image,detections,height,width,labels):
    for i in range(10):
        if detections['detection_scores'][0][i].numpy()>0.5:
            ymin = int(detections['detection_boxes'][0][i][0].numpy() * height)
            xmin = int(detections['detection_boxes'][0][i][1].numpy() * width)
            ymax = int(detections['detection_boxes'][0][i][2].numpy() * height)
            xmax = int(detections['detection_boxes'][0][i][3].numpy() * width)
            label_location=detections['detection_classes'][0][i].numpy()+1
            label_name=labels[label_location]['name']
            score=int(detections['detection_scores'][0][i].numpy()*100)+1
            cv2.rectangle(image_np_with_detections, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 2)
            cv2.rectangle(image_np_with_detections, (int(xmin), int(ymin-17)), (int(xmin+140), int(ymin)), (0, 0, 0), -1)
            cv2.putText(image_np_with_detections, str(label_name)+':', (int(xmin), int(ymin - 3)),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(image_np_with_detections, str(score)+'%', (int(xmin+90), int(ymin-3)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1,
                        cv2.LINE_AA)
    return image
def extract_unseen_person(used_ids,detections_track):
    new_center_points = {}
    for used in used_ids:
        center = detections_track[used]
        new_center_points[used] = center
    detections_track = new_center_points.copy()
    return detections_track
def check_new_person(counts,boxes,detections,detections_track,used_ids,person_in_count):
    for i in range(10):
        if detections['detection_scores'][0][i].numpy()>0.5:
            ymin = int(detections['detection_boxes'][0][i][0].numpy() * height)
            xmin = int(detections['detection_boxes'][0][i][1].numpy() * width)
            ymax = int(detections['detection_boxes'][0][i][2].numpy() * height)
            xmax = int(detections['detection_boxes'][0][i][3].numpy() * width)

            (x, y, w, h) = xmin, ymin, abs(xmin - xmax), abs(ymin - ymax)
            if 260<x<330:
                tracker = opencv_trackers[trackerName]()
                trackers.add(tracker, ROI_frame, (x, y, w, h))
                for coordinates in boxes:
                    x, y, w, h = coordinates
                    cx = (x + x + w) // 2
                    cy = (y + y + h) // 2
                    person_in = False
                    person_detected_before = False
                    for counted, centers in detections_track.items():
                        euclidean_dist = math.hypot(cx - centers[0], cy - centers[1])
                        if euclidean_dist < 150:
                            detections_track[counted] = (cx, cy)
                            used_ids.append(counted)
                            person_detected_before = True
                            break
                    if person_detected_before is False:
                        if x > 300:
                            person_in = True
                        if person_in is True:
                            person_in_count += 1
                        detections_track[counts] = (cx, cy)
                        used_ids.append(counts)
                        counts += 1
    return counts,used_ids,detections_track,person_in_count
labels = label_map_util.create_category_index_from_labelmap(path_to_labels,use_display_name=True)
cap = cv2.VideoCapture('gettyimages-85314342-640_adpp.mp4') # used to capture video
trackerName = 'csrt'
opencv_trackers = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
}
trackers = cv2.legacy.MultiTracker_create()
counts=0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
thickness = 1
detections_track= {}
person_in_count=0

while True:
    ret, image_np = cap.read() # Reads captured video returns to true and video
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32) #conversion to tensor
    #Tensorflow reads the images as float32 type arrays so we convert the input with above code
    # np.expand_dims convert the image to numbers in array as RGB code.
    detections = detection_process(input_tensor)
    image_np_with_detections = image_np.copy()
    used_ids = []
    height, width = image_np_with_detections.shape[:2]
    ROI_frame = image_np_with_detections
    (ret1, boxes) = trackers.update(ROI_frame)
    trackers.clear()
    trackers = cv2.legacy.MultiTracker_create()
    counts,used_ids,detections_track,person_in_count=check_new_person(counts,boxes,detections,detections_track,used_ids,person_in_count)
    detections_track=extract_unseen_person(used_ids,detections_track)
    image_np_with_detections=draw_rectangle_boxes(image_np_with_detections,detections,height,width,labels)
    cv2.putText(image_np_with_detections, str(counts-person_in_count), (590,325), font,1, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(image_np_with_detections, str(person_in_count), (590,345), font, 1, (255, 255, 255), thickness,cv2.LINE_AA)
    cv2.putText(image_np_with_detections, 'People in:', (440, 325), font,1, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(image_np_with_detections, 'People out:', (440, 345), font, 1, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(image_np_with_detections, 'ROI:', (260, 50), font, 1, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.rectangle(image_np_with_detections, (260, 0), (330, int(height)), (255, 0, 0), 1)

    cv2.imshow('object detection',cv2.resize(image_np_with_detections, (int(width), int(height))))  # cv2 code to display video.

    if cv2.waitKey(1) & 0xFF == ord('q'):  # number inside waitkey indicates the number of miliseconds that video will be displayed
        break
cap.release()
cv2.destroyAllWindows()