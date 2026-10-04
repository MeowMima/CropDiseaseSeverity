from pathlib import Path

import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------
# MODEL PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "best (1).pt"


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

yolo_model = YOLO(str(YOLO_MODEL_PATH))


# ---------------------------------------------------------
# DETECTION FUNCTION
# ---------------------------------------------------------

def detect_crop(image: np.ndarray):
    """
    Run YOLO detection on an image and return
    the highest-confidence detection.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format.

    Returns
    -------
    dict
        Detection information containing:
        - class_id
        - class_name
        - confidence
        - bbox
    """

    results = yolo_model(image, verbose=False)

    if not results:
        return None

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes

    # Confidence values
    confidences = boxes.conf.cpu().numpy()

    # Highest-confidence detection
    best_index = int(np.argmax(confidences))

    confidence = float(confidences[best_index])

    class_id = int(
        boxes.cls[best_index].cpu().numpy()
    )

    bbox = (
        boxes.xyxy[best_index]
        .cpu()
        .numpy()
        .astype(int)
        .tolist()
    )

    # Get class name from YOLO model
    class_names = yolo_model.names

    if isinstance(class_names, dict):
        class_name = class_names.get(
            class_id,
            str(class_id)
        )
    else:
        class_name = class_names[class_id]

    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox,
    }