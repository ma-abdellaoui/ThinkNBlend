from transformers import pipeline
from PIL import Image
from think_n_blend.config import OBJECT_DETECTION_MODEL
from think_n_blend.schemas import BoundingBox

def detect_reference_object(image_path: str, reference_object_label: str) -> BoundingBox | None:
    """
    Detects the reference object in the main image using a zero-shot object detection model.
    """
    detector = pipeline(model=OBJECT_DETECTION_MODEL, task="zero-shot-object-detection")
    image = Image.open(image_path)
    
    predictions = detector(image, candidate_labels=[reference_object_label])

    best_box = None
    max_score = -1.0
    for prediction in predictions:
        if prediction['score'] > max_score:
            max_score = prediction['score']
            best_box = prediction['box']
    
    if best_box:
        return (best_box['xmin'], best_box['ymin'], best_box['xmax'], best_box['ymax'])
    
    return None
