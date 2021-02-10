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
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
# from IPython.display import display

from object_detection.utils import ops as utils_ops, np_box_list_ops, np_box_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import np_box_list

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


def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # x_min, y_min, x_max, y_max
    cajas = []
    for k, j, l in zip(output_dict['detection_boxes'],
                       output_dict['detection_classes'],
                       output_dict['detection_scores']):
        if l > threshold:
            if j in [3, 6, 8]:
                cajas.append([k[1] * width, k[0] * height, k[3] * width, k[2] * height])

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    # display(Image.fromarray(image_np))
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.savefig("outputs/detection_output{}.png".format(i))
    return cajas


# ################# Main Program ##################
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Input of images
# If you want to test the code with your images, just add path of the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
IMAGE_SIZE = (12, 8)
width = 1200
height = 800
threshold = 0.5

# Detection model
MODEL_NAME = 'efficientdet_d0_coco17_tpu-32'
detection_model = load_model(MODEL_NAME)

# Box outputs
i = 1

for image_path in TEST_IMAGE_PATHS:
    print("Imagen ", i)
    cajas = show_inference(detection_model, image_path)
    # Deteccion de huecos
    cajas.sort(key=lambda y: y[0])
    solucion = []
    im = Image.open("outputs/detection_output{}.png".format(i))
    draw = ImageDraw.Draw(im)
    line_color = (0, 0, 255)  # Azul
    # Si no  hay vehículos, hay hueco seguro!
    if len(cajas) == 0:
        solucion.append([0, width])
        draw.line([(200, 400), (width, 400)], fill=line_color, width=3)
    else:
        # Si solo hay un vehiculo
        huecos = []  # x_min, x_max, long coche1, long coche2
        if cajas[0][0] > 0:
            huecos.append([0, cajas[0][0], cajas[0][2] - cajas[0][0]])
        for x in range(0, len(cajas) - 1):
            if cajas[x][2] < cajas[x + 1][0]:
                huecos.append(
                    [cajas[x][2], cajas[x + 1][0], cajas[x][2] - cajas[x][0], cajas[x + 1][2] - cajas[x + 1][0]])
        if cajas[len(cajas) - 1][2] < width:
            huecos.append([cajas[len(cajas) - 1][2], width, cajas[len(cajas) - 1][2] - cajas[len(cajas) - 1][0]])
        # Vemos cuáles de esos huecos son válidos
        for x in range(0, len(huecos)):
            size_hueco = huecos[x][1] - huecos[x][0]
            if len(huecos[x]) == 3:
                if huecos[x][0] == 0:
                    if size_hueco >= (huecos[x][2] / 3):
                        solucion.append([huecos[x][0], huecos[x][1]])
                        draw.line([(200, 400), (huecos[x][1], 400)], fill=line_color, width=3)
                elif huecos[x][1] == width:
                    if size_hueco >= (5 / 3) * huecos[x][2]:
                        solucion.append([huecos[x][0], huecos[x][1]])
                        draw.line([(huecos[x][0], 400), (width, 400)], fill=line_color, width=3)
            else:
                if size_hueco >= (huecos[x][2] + huecos[x][3]) / 6:
                    solucion.append([huecos[x][0], huecos[x][1]])
                    draw.line([(huecos[x][0], 400), (huecos[x][1], 400)], fill=line_color, width=3)

    im.save("outputs/detection_output_hueco{}.png".format(i))
    i = i + 1
    print("Nº Huecos: ", len(solucion))
    if len(solucion) > 0:
        print(solucion)
