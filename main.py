import pathlib

import numpy as np
# import os
# import six.moves.urllib as urllib
# import sys
# import tarfile
import tensorflow as tf
# import zipfile

# from collections import defaultdict
# from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
# from IPython.display import display

from object_detection.utils import ops as utils_ops
# from object_detection.utils import np_box_list_ops, np_box_ops
from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util
# from object_detection.utils import np_box_list

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# Methods
def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir) / "saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, path_image):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(path_image))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # x_min, y_min, x_max, y_max
    box_filter = []
    for k, j, l in zip(output_dict['detection_boxes'],
                       output_dict['detection_classes'],
                       output_dict['detection_scores']):
        if l > threshold:
            if j in [3, 6, 8]:
                box_filter.append([k[1] * width, k[0] * height, k[3] * width, k[2] * height])

    return box_filter


# ################# Main Program ##################
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Input of images
# If you want to test the code with your images, just add path of the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))


# Detection model
MODEL_NAME = 'efficientdet_d0_coco17_tpu-32'
detection_model = load_model(MODEL_NAME)
threshold = 0.5
# Box outputs
i = 1

for image_path in TEST_IMAGE_PATHS:
    im = Image.open(image_path)
    width, height = im.size
    draw = ImageDraw.Draw(im)
    car_color = (0, 0, 255)  # Blue
    gap_color = (255, 0, 0)  # Red

    print("Image ", i)
    boxes = show_inference(detection_model, image_path)

    # Gap detection
    boxes.sort(key=lambda y: y[0])
    solution = []

    # If there is no vehicle, there is a gap
    if len(boxes) == 0:
        solution.append([0, width])
        draw.line([(0, height / 2), (width, height / 2)], fill=gap_color, width=5)
    else:
        gaps = []  # x_min, x_max, long car1, long car2
        if boxes[0][0] > 0:
            gaps.append([0, boxes[0][0], boxes[0][2] - boxes[0][0]])
        for x in range(0, len(boxes) - 1):
            x_min, y_min, x_max, y_max = boxes[x]
            draw.line([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)], fill=car_color,
                      width=3)
            if boxes[x][2] < boxes[x + 1][0]:
                gaps.append(
                    [boxes[x][2], boxes[x + 1][0], boxes[x][2] - boxes[x][0], boxes[x + 1][2] - boxes[x + 1][0]])
        x_min, y_min, x_max, y_max = boxes[len(boxes) - 1]
        draw.line([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)], fill=car_color,
                  width=3)
        if boxes[len(boxes) - 1][2] < width:
            gaps.append([boxes[len(boxes) - 1][2], width, boxes[len(boxes) - 1][2] - boxes[len(boxes) - 1][0]])
        # Filter only valid gaps
        for x in range(0, len(gaps)):
            gap_size = gaps[x][1] - gaps[x][0]
            if len(gaps[x]) == 3:
                if gaps[x][0] == 0:
                    if gap_size >= (gaps[x][2] / 3):
                        solution.append([gaps[x][0], gaps[x][1]])
                        draw.line([(0, height / 2), (gaps[x][1], height / 2)], fill=gap_color, width=3)
                elif gaps[x][1] == width:
                    if gap_size >= (5 / 3) * gaps[x][2]:
                        solution.append([gaps[x][0], gaps[x][1]])
                        draw.line([(gaps[x][0], height / 2), (width, height / 2)], fill=gap_color, width=3)
            else:
                if gap_size >= (gaps[x][2] + gaps[x][3]) / 6:
                    solution.append([gaps[x][0], gaps[x][1]])
                    draw.line([(gaps[x][0], height / 2), (gaps[x][1], height / 2)], fill=gap_color, width=3)

    im.save("outputs/detection_output{}.png".format(i))
    i = i + 1
