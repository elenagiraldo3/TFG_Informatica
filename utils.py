import pathlib

import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import ops as utils_ops


# Methods
def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True
    )

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
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'], image.shape[0], image.shape[1]
        )
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, path_image):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(path_image))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    return output_dict


def filter_vehicles(output_dict, width, height, threshold):
    # x_min, y_min, x_max, y_max
    boxes = list()
    for detection_box, detection_class, detection_score in zip(output_dict['detection_boxes'],
                                                               output_dict['detection_classes'],
                                                               output_dict['detection_scores']):
        if detection_score > threshold:
            if detection_class in [3, 6, 8]:
                boxes.append([detection_box[1] * width,
                              detection_box[0] * height,
                              detection_box[3] * width,
                              detection_box[2] * height])

    boxes.sort(key=lambda y: y[0])
    return boxes


def gap_detection(boxes, width, draw, car_color):
    gaps = list()
    # If there is no vehicle, there is a gap
    if not boxes:
        gaps.append((0, width))
    else:
        gaps = list()  # x_min, x_max, long car1, long car2
        if boxes[0][0] > 0:
            gaps.append((0, boxes[0][0], boxes[0][2] - boxes[0][0]))
        for x in range(0, len(boxes) - 1):
            x_min, y_min, x_max, y_max = boxes[x]
            draw.line(
                [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)],
                fill=car_color,
                width=5
            )
            if boxes[x][2] < boxes[x + 1][0]:
                gaps.append(
                    [boxes[x][2], boxes[x + 1][0], boxes[x][2] - boxes[x][0], boxes[x + 1][2] - boxes[x + 1][0]]
                )
        x_min, y_min, x_max, y_max = boxes[len(boxes) - 1]
        draw.line([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)], fill=car_color,
                  width=5)
        if boxes[len(boxes) - 1][2] < width:
            gaps.append([boxes[len(boxes) - 1][2], width, boxes[len(boxes) - 1][2] - boxes[len(boxes) - 1][0]])
    return gaps


def valid_gaps(gaps, height, width, draw, gap_color):
    solution = list()
    for gap in range(0, len(gaps)):
        gap_size = gaps[gap][1] - gaps[gap][0]
        if len(gaps[gap]) == 2:
            draw.line([(0, height / 2), (width, height / 2)], fill=gap_color, width=5)
            solution.append([gaps[gap][0], gaps[gap][1]])
        elif len(gaps[gap]) == 3:
            if gaps[gap][0] == 0:
                if gap_size >= (gaps[gap][2] / 3):
                    solution.append([gaps[gap][0], gaps[gap][1]])
                    draw.line([(0, height / 2), (gaps[gap][1], height / 2)], fill=gap_color, width=5)
            elif gaps[gap][1] == width:
                if gap_size >= (5 / 3) * gaps[gap][2]:
                    solution.append([gaps[gap][0], gaps[gap][1]])
                    draw.line([(gaps[gap][0], height / 2), (width, height / 2)], fill=gap_color, width=5)
        else:
            if gap_size >= (gaps[gap][2] + gaps[gap][3]) / 6:
                solution.append([gaps[gap][0], gaps[gap][1]])
                draw.line([(gaps[gap][0], height / 2), (gaps[gap][1], height / 2)], fill=gap_color, width=5)
    return solution
