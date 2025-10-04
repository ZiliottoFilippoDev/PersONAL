# typing
from typing import List, Dict

import numpy as np
# inference
from inference.models import YOLOWorld

# cv2
import cv2

# supervision
import supervision as sv


class YOLOWorldDetector:
    def __init__(self,
                 confidence_threshold: float
                 ):
        self.model = YOLOWorld(model_id="yolo_world/l")
        self.confidence_threshold = confidence_threshold
        self.classes = None

    def set_classes(self,
                    classes: List[str]
                    ):
        self.classes = classes
        self.model.set_classes(classes)

    def detect(self, image: np.ndarray) -> dict:
        if self.classes is None:
            raise ValueError("Classes must be set before detecting")

        results = self.model.infer(image, confidence=self.confidence_threshold)

        preds = {
            "boxes": [],
            "scores": []
        }

        for detection in results.predictions:
            cls = detection.class_id
            class_name = detection.class_name

            if class_name == self.classes[0] and detection.confidence > self.confidence_threshold:
                x1 = detection.x - detection.width / 2
                y1 = detection.y - detection.height / 2
                x2 = detection.x + detection.width / 2
                y2 = detection.y + detection.height / 2

                # Check if box is not a point
                if x1 != x2 and y1 != y2:
                    preds["boxes"].append([x1, y1, x2, y2])
                    preds["scores"].append(detection.confidence)

        return preds

if __name__ == "__main__":
    # Test the YOLO World Detector
    detector = YOLOWorldDetector(confidence_threshold=0.5)
    detector.set_classes(["person", "car", "truck", "bus", "bicycle", "motorbike", "traffic light", "stop sign"])

    # Load an image
    image = cv2.imread("test_images/a.jpg")

    # Detect objects in the image
    detections = detector.detect(image)

    # Display the image with the detections
    image_with_detections = detections.draw_on_image(image)
    cv2.imshow("Detections", image_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
