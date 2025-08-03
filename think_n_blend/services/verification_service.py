from transformers import pipeline
from PIL import Image
import easyocr
from think_n_blend.schemas import VerificationResult

def verify_object_insertion(image_path: str, expected_object: str) -> VerificationResult:
    """
    Verifies that an object was successfully inserted using object detection.
    """
    try:
        detector = pipeline(model="google/owlv2-base-patch16-ensemble", task="zero-shot-object-detection")
        image = Image.open(image_path)
        
        predictions = detector(image, candidate_labels=[expected_object])
        
        if predictions:
            best_prediction = max(predictions, key=lambda x: x['score'])
            return VerificationResult(
                object_detected=best_prediction['score'] > 0.5,
                text_detected=False,
                object_confidence=best_prediction['score'],
                detected_objects=[best_prediction]
            )
        else:
            return VerificationResult(
                object_detected=False,
                text_detected=False,
                object_confidence=0.0
            )
    except Exception as e:
        return VerificationResult(
            object_detected=False,
            text_detected=False,
            error_message=str(e)
        )

def verify_text_insertion(image_path: str, expected_text: str) -> VerificationResult:
    """
    Verifies that text was successfully inserted using OCR.
    """
    try:
        # Initialize EasyOCR
        reader = easyocr.Reader(['en'])
        
        # Read the image
        results = reader.readtext(image_path)
        
        detected_texts = []
        for (bbox, text, confidence) in results:
            detected_texts.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
        
        # Check if expected text is found
        text_found = any(expected_text.lower() in text['text'].lower() for text in detected_texts)
        max_confidence = max([text['confidence'] for text in detected_texts]) if detected_texts else 0.0
        
        return VerificationResult(
            object_detected=False,
            text_detected=text_found,
            text_confidence=max_confidence,
            detected_text=" ".join([text['text'] for text in detected_texts])
        )
    except Exception as e:
        return VerificationResult(
            object_detected=False,
            text_detected=False,
            error_message=str(e)
        )

def verify_insertion_quality(image_path: str, insertion_type: str, expected_content: str) -> VerificationResult:
    """
    Verifies the quality of an insertion based on type.
    """
    if insertion_type == "object":
        return verify_object_insertion(image_path, expected_content)
    elif insertion_type == "text":
        return verify_text_insertion(image_path, expected_content)
    else:
        raise ValueError(f"Unknown insertion type: {insertion_type}") 