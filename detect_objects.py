import argparse
from pathlib import Path
import json
import logging
import sys

from yoloobjectdetector.yoloobjectdetector import YoloObjectDetector 

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


if __name__ == "__main__":

    # Read config file
    with open('resources/config.json', 'r') as f:
        config = json.load(f)

    weights_path = Path(config["weights_path"])
    yolo_cfg_path = config["yolo_cfg_path"]
    generated_files_folder = Path(config["generated_files_folder"])
    detection_threshold = config["detection_threshold"]
    detectable_classes_path = config["detectable_classes_path"]

    with open(detectable_classes_path, "r") as f:
        detectable_classes = [line.strip() for line in f.readlines()]

    # Read arguments
    parser = argparse.ArgumentParser(prog='yolo_object_detector')
    parser.add_argument('-i', '--input-video-path', help='Path of the video to process', required=True, type=str)
    parser.add_argument('-c', '--classes-to-detect', help='Object class to detect, availables classes can be found in resources/coco.names', required=True, action='append', type=str)
    parser.add_argument('-d', '--display-class-name', help='Will display the name of the class if each detected entity', required=False, default=False, type=bool)
    parser.add_argument('-t', '--detection-threshold', help='Confidence threshold above which an object should be localized in the video', required=False, default=detection_threshold, type=float)
    args = vars(parser.parse_args())

    input_video_path = Path(args['input_video_path'])
    classes_to_detect = args["classes_to_detect"]
    display_class_name = args["display_class_name"]
    detection_threshold = args["detection_threshold"]

    # Check yolo weights have been downloaded
    if not weights_path.is_file():
        logging.error(f'{weights_path} does not exists, you can download the weight file by running wget -O resources/yolov3.weights https://pjreddie.com/media/files/yolov3.weights')
        sys.exit()

    # Check input video path
    if not input_video_path.is_file():
        logging.error(f'{input_video_path} does not exists')
        sys.exit()

    # Check class is in coco dataset
    for c in classes_to_detect:
        if c not in detectable_classes:
            logging.error(f'"{c}" can not be detected by this model, you can check detectable classes in resources/coco.names')
            sys.exit()

    # Check threshold is between 0 and 1
    if detection_threshold < 0 or detection_threshold > 1:
        logging.error(f'detection-threshold should be beetween 0 and 1, passed value is {detection_threshold}')
        sys.exit()

    # Generate path of processed video
    output_video_path = (generated_files_folder / f'{input_video_path.stem}_yolo_processed.avi').absolute()

    # Instantiate model
    detector = YoloObjectDetector(weights_path, yolo_cfg_path, detectable_classes)

    # Detect, localize and draw boxes around objects in video 
    detector.detect_objects_in_video(input_video_path, output_video_path, classes_to_detect, display_class_name, detection_threshold)

    logging.info(f"Video generated: {output_video_path}")


