from dataclasses import dataclass
from typing import Literal, Tuple, Optional

RelativePosition = Literal["top", "bottom", "left", "right"]
BoundingBox = Tuple[int, int, int, int]

@dataclass
class ReferenceObject:
    label: str
    description: str
    position_role: str = "reference"

@dataclass
class TargetObject:
    label: str
    description: str
    relative_position: RelativePosition
    inpainting_description: str

@dataclass
class TextInsertion:
    text: str
    font_size: int = 48
    font_color: str = "white"
    background_color: Optional[str] = None
    position: RelativePosition = "top"
    inpainting_description: str = ""

@dataclass
class Gpt4VisionResponse:
    reference_object: ReferenceObject
    target_object: TargetObject

@dataclass
class InsertionResult:
    success: bool
    output_path: str
    bounding_box: Optional[BoundingBox] = None
    confidence_score: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class VerificationResult:
    object_detected: bool
    text_detected: bool
    object_confidence: Optional[float] = None
    text_confidence: Optional[float] = None
    detected_text: Optional[str] = None
    detected_objects: Optional[list] = None

