import os


model_name='model_person'
#first we take the path of the necessary folders.
model_direction=os.path.join(os.getcwd(),os.path.join(model_name,'saved_model'))
path_to_config=os.path.join(os.getcwd(),os.path.join(model_name,'pipeline.config'))
path_to_labels=os.path.join(os.getcwd(),'label_map.pbtxt')
path_to_checkpoint=os.path.join(os.getcwd(),os.path.join(model_name,'checkpoint/'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

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
labels = label_map_util.create_category_index_from_labelmap(path_to_labels,
                                                                    use_display_name=True)

import cv2

cap = cv2.VideoCapture(0) # used to capture video

import numpy as np

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
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)


    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600))) #cv2 code to display video.

    if cv2.waitKey(1) & 0xFF == ord('q'):  # number inside waitkey indicates the number of miliseconds that video will be displayed
        break

cap.release()
cv2.destroyAllWindows()