import cv2
import numpy as np
from tqdm import tqdm


class YoloObjectDetector:
    def __init__(self, weights_path, yolo_cfg_path, detectable_classes):

        # Instantiate yolo model
        self.model = cv2.dnn.readNet(str(weights_path), yolo_cfg_path)
        self.layer_names = self.model.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]

        # Classes detected by yolo (which are coco dataset classes)
        self.detectable_classes = detectable_classes

        # Colors used to differenciate objects
        self.colors = np.random.uniform(0, 255, size=len(self.detectable_classes))

    def detect_objects_in_frame(self, frame, classes_to_detect, display_class_name, detection_threshold):

        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.model.setInput(blob)

        # Put frame through the model to detect objects
        outs = self.model.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # Loop through all detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                label = str(self.detectable_classes[class_id])

                # Filter out objects with low detection threshold and from useless classes
                if (confidence > detection_threshold) & (label in classes_to_detect):

                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Keep only one box by object witth "Non maximum supression" algorithm
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Add boxes of different colors around object detected on the frame
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.detectable_classes[class_ids[i]])

                color = self.colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                if display_class_name:
                    cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

        return frame


    def detect_objects_in_video(self, input_video_path, output_video_path, classes_to_detect, display_class_name, detection_threshold):

        # Read video
        cap = cv2.VideoCapture(str(input_video_path))

        # Get video informations
        f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Instantiate object to save frames and create a video 
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (f_width, f_height))

        #length = 20
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0

        # Instantiate progress bar
        pbar = tqdm(total = length)

        while(cap.isOpened() & (frame_count < length)):
            frame_count +=1

            # Read frame
            ret, frame = cap.read()

            if ret:
                frame = self.detect_objects_in_frame(frame, classes_to_detect, display_class_name, detection_threshold)

                # Save frame
                out.write(frame)

                # Update progress bar
                pbar.update(1)


        cap.release()
        out.release()

