"""MIT License

Copyright (c) 2021 Elena Giraldo del Viejo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import os
import pathlib
import time
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
    start_time = time.time()
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # 1. Detection model
    MODEL_NAME = 'efficientdet_d0_coco17_tpu-32'
    # MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
    # MODEL_NAME = 'efficientdet_d7_coco17_tpu-32'
    detection_model = load_model(MODEL_NAME)

    # 2. Input of images
    # If you want to test the code with your images, just add path of the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path('images')
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

    threshold = 0.5
    cont_vehicles = 0
    cont_gaps = 0
    for i, image_path in enumerate(TEST_IMAGE_PATHS):
        im = Image.open(image_path)
        width, height = im.size
        draw = ImageDraw.Draw(im)
        car_color = (0, 255, 255)  # Yellow
        gap_color = (248, 0, 0)  # Red

        # 3. Object detection
        objects = show_inference(detection_model, image_path)

        # 4. Filter vehicles
        boxes = filter_vehicles(
            output_dict=objects,
            width=width,
            height=height,
            threshold=threshold
        )
        cont_vehicles += len(boxes)

        # 5. Gap detection
        gaps = gap_detection(
            boxes=boxes,
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
        cont_gaps += len(solution)

        # 7. Output image
        string_path = os.path.basename(image_path)
        print(f"Image: {string_path}")
        print(f"Number of vehicles: {len(boxes)}")
        print(f"Number of gaps: {len(solution)}")
        im.save(f"outputs/{string_path}")

    print(f"Number of totals vehicles: {cont_vehicles}")
    print(f"Number of totals gaps: {cont_gaps}")
    print("--- %s seconds ---" % (time.time() - start_time))
