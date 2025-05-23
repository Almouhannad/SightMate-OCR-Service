import numpy as np
import cv2
import supervision as sv
from typing import List, Tuple

class ImageAnnotator:
    def __init__(self, color: sv.Color = sv.Color.BLACK, thickness: int = 2):
        self.box_annotator = sv.BoxAnnotator(
            color=color,
            thickness=thickness
        )

    def annotate(self, image_bytes: bytes, boxes: List[List[Tuple[float, float]]]) -> bytes:
        """
        Annotate image with bounding boxes using supervision.
        
        Args:
            image_bytes: Raw image bytes
            boxes: List of boxes, where each box is a list of (x,y) coordinates
            
        Returns:
            Annotated image as bytes
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert boxes to supervision format (xyxy)
        xyxy = []
        for box in boxes:
            x_min = int(min([point[0] for point in box]))
            y_min = int(min([point[1] for point in box]))
            x_max = int(max([point[0] for point in box]))
            y_max = int(max([point[1] for point in box]))
            xyxy.append([x_min, y_min, x_max, y_max])

        # Create supervision detections
        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.ones(len(xyxy)), # Mock values
            class_id=np.zeros(len(xyxy))
        )

        # Annotate image
        annotated_image = self.box_annotator.annotate(
            scene=image,
            detections=detections
        )

        # Convert back to bytes
        _, buffer = cv2.imencode('.png', annotated_image)
        return buffer.tobytes() 