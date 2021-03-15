import os
import pathlib

import tensorflow as tf
from PIL import Image, ImageDraw
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

from utils import show_inference, load_model, filter_vehicles, gap_detection, valid_gaps

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

if __name__ == "__main__":
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # 1. Detection model
    MODEL_NAME = 'efficientdet_d0_coco17_tpu-32'
    detection_model = load_model(MODEL_NAME)

    # 2. Input of images
    # If you want to test the code with your images, just add path of the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path('images')
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

    threshold = 0.5
    for i, image_path in enumerate(TEST_IMAGE_PATHS):
        im = Image.open(image_path)
        width, height = im.size
        draw = ImageDraw.Draw(im)
        car_color = (0, 0, 255)  # Blue
        gap_color = (255, 0, 0)  # Red

        # 3. Object detection
        objects = show_inference(detection_model, image_path)

        # 4. Filter vehicles
        boxes = filter_vehicles(
            output_dict=objects,
            width=width,
            height=height,
            threshold=threshold
        )

        # 5. Gap detection
        gaps = gap_detection(
            boxes=boxes,
            height=height,
            width=width,
            draw=draw,
            car_color=car_color
        )

        # 6. Filter only valid gaps
        solution = valid_gaps(
            gaps=gaps,
            height=height,
            width=width,
            draw=draw,
            gap_color=gap_color
        )

        # 7. Output image
        string_path = os.path.basename(image_path)
        print(string_path)
        print(f"Image {i}")
        print(f"Number of gaps: {len(solution)}")
        for n in range(1, len(solution) + 1):
            print(f"Gap {n}: {solution[n - 1]}")
        im.save(f"outputs/{string_path}")
