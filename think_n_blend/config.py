GPT4_VISION_PROMPT = """
You are a vision model assistant. You are given:
- A main image showing a real-world scene.
- A cropped image of an object (e.g., a cap, bottle, book, etc.).

Your task is to analyze both images and decide where in the main image the object (from the crop) could be realistically placed.

You must:
Identify a plausible reference object already present in the main image.
Determine a relative position where the new object could naturally fit. Valid positions are: "top", "bottom", "left", or "right" — relative to the reference object's bounding box.
Generate a concise, inpainting-style description (like a Stable Diffusion prompt) that clearly describes what the final image should look like after placing the object in context.

Output the result in this exact JSON format:
{
  "reference_object": {
    "label": "object_label_in_main_image",
    "description": "Short explanation of the reference object and why it's suitable.",
    "position_role": "reference"
  },
  "target_object": {
    "label": "object_label_from_crop",
    "description": "Short explanation of what the object is and why it's placed here.",
    "relative_position": "top",
    "inpainting_description": "Short, high-quality prompt describing the object after placement for an inpainting model"
  }
}
"""

GPT4_TEXT_VISION_PROMPT = """You are a vision model assistant. You are given:
- A main image showing a real-world scene.
- A text string that needs to be inserted into the image.

Your task is to analyze the main image and decide where the text could be realistically placed.

You must:
1. Identify a plausible reference object already present in the main image.
2. Determine a relative position where the text could naturally fit.
   Valid positions are: "top", "bottom", "left", or "right" — relative to the reference object's bounding box.
3. Generate a concise, inpainting-style description that clearly describes what the final image should look like after placing the text in context.

Output the result in this exact JSON format:
{{
  "reference_object": {{
    "label": "object_label_in_main_image",
    "description": "Short explanation of the reference object and why it's suitable.",
    "position_role": "reference"
  }},
  "target_object": {{
    "label": "text_label",
    "description": "Short explanation of what the text is and why it's placed here.",
    "relative_position": "top",
    "inpainting_description": "Short, high-quality prompt describing the text after placement for an inpainting model"
  }}
}}

The text to insert is: "{text}"
"""

GPT4_VISION_MODEL = "gpt-4o"
OBJECT_DETECTION_MODEL = "google/owlv2-base-patch16-ensemble"

# Submodule paths
SUBMODULES_DIR = "submodules"
UNICOMBINE_PATH = f"{SUBMODULES_DIR}/UniCombine"

# Model configurations
DIFFUSION_MODELS = {
    "unicombine": {
        "path": UNICOMBINE_PATH,
        "inference_script": "inference.py",
        "requirements": "requirements.txt",
        "description": "UniCombine for object and text insertion"
    },
    "simple_paste": {
        "path": None,
        "inference_script": None,
        "requirements": None,
        "description": "Simple image pasting (no diffusion model required)"
    }
}

OBJECT_DETECTION_MODELS = {
    "owlv2": {
        "model": "google/owlv2-base-patch16-ensemble",
        "task": "zero-shot-object-detection",
        "description": "OWLv2 for zero-shot object detection"
    }
}

# Default model selections
DEFAULT_DIFFUSION_MODEL = "unicombine"
DEFAULT_OBJECT_DETECTION_MODEL = "owlv2"

# Processing configurations
DEFAULT_BATCH_SIZE = 1
DEFAULT_VERIFY_INSERTIONS = True
DEFAULT_SAVE_INTERMEDIATE_RESULTS = False
SKIP_DIFFUSION_MODEL = False  # Flag to skip diffusion model and use simple pasting

# Output configurations
DEFAULT_OUTPUT_FORMAT = "jpg"
DEFAULT_COMPRESSION_QUALITY = 95

