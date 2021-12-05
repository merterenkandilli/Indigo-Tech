import os

import numpy as np

model_name='model_person'
#first we take the path of the necessary folders.
model_direction=os.path.join(os.getcwd(),os.path.join(model_name,'saved_model'))
path_to_config=os.path.join(os.getcwd(),os.path.join(model_name,'pipeline.config'))
path_to_labels=os.path.join(os.getcwd(),'label_map.pbtxt')
path_to_checkpoint=os.path.join(os.getcwd(),os.path.join(model_name,'checkpoint/'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
# importing necessary files.
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

configurations=config_util.get_configs_from_pipeline_file(path_to_config) #reads entire config file.
model_configuration=configurations['model']  #reads the code in model{ .. }
detection_model = model_builder.build(model_config=model_configuration, is_training=False)
#as far as I understand model_builder.build constructs an flow chart for detection.

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(path_to_checkpoint, 'ckpt-0')).expect_partial()
path_to_image=os.path.join(os.getcwd(),'vesikalik.jpg')
image_test=cv2.imread(path_to_image)
category_index = label_map_util.create_category_index_from_labelmap(path_to_labels,
                                                                    use_display_name=True)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_test, 0), dtype=tf.float32)
#it takes the picture as array and its dimensions (ex. 560x560)
image,shapes=detection_model.preprocess(input_tensor)
prediction_dict = detection_model.predict(image, shapes) # it is used to compute class probabilities
detections = detection_model.postprocess(prediction_dict, shapes)
image_with_detections=image_test.copy()
label_id_offset=1
viz_utils.visualize_boxes_and_labels_on_image_array(
          image_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

cv2.imshow('object detection', cv2.resize(image_with_detections, (800, 600)))
cv2.waitKey(0)