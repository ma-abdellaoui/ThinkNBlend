import base64
from pathlib import Path
from PIL import Image, ImageDraw

def encode_image(image_path: str) -> str:
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_mask_from_box(image_path: str, box: tuple[int, int, int, int], output_path: str) -> str:
    """Creates a mask image from a bounding box."""
    image = Image.open(image_path)
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box, fill=255)
    mask.save(output_path)
    return output_path

def create_dummy_image(path: str, size: tuple[int, int], color: str):
    """Creates a dummy image file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color=color).save(path)

def save_bounding_box_visualization(
    image_path: str,
    reference_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
    output_path: str,
):
    """Saves an image with the reference and target bounding boxes drawn on it."""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    draw.rectangle(reference_box, outline="red", width=3)
    draw.rectangle(target_box, outline="green", width=3)
    image.save(output_path)
