import os
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
from think_n_blend.schemas import InsertionResult
from think_n_blend.utils.image_utils import create_mask_from_box

def resize_object_to_fit_box(object_image: Image.Image, target_box: Tuple[int, int, int, int]) -> Image.Image:
    """Resize object image to fit the target bounding box."""
    x1, y1, x2, y2 = target_box
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Resize object to fit the box while maintaining aspect ratio
    object_width, object_height = object_image.size
    aspect_ratio = object_width / object_height
    box_aspect_ratio = box_width / box_height
    
    if aspect_ratio > box_aspect_ratio:
        # Object is wider than box, fit to width
        new_width = box_width
        new_height = int(box_width / aspect_ratio)
    else:
        # Object is taller than box, fit to height
        new_height = box_height
        new_width = int(box_height * aspect_ratio)
    
    return object_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def create_text_image_for_box(text: str, target_box: Tuple[int, int, int, int], font_size: int = 48, font_color: str = "white") -> Image.Image:
    """Create a text image that fits the target bounding box with transparent background."""
    x1, y1, x2, y2 = target_box
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Create transparent image the size of the target box
    text_image = Image.new('RGBA', (box_width, box_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_image)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position to center it in the box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width_actual = bbox[2] - bbox[0]
    text_height_actual = bbox[3] - bbox[1]
    
    # Center the text in the box
    x = (box_width - text_width_actual) // 2
    y = (box_height - text_height_actual) // 2
    
    # Draw text with specified color
    draw.text((x, y), text, fill=font_color, font=font)
    
    return text_image

def simple_object_paste(
    main_image_path: str,
    object_crop_path: str,
    target_box: Tuple[int, int, int, int],
    output_path: str = None,
) -> InsertionResult:
    """
    Simple object pasting without diffusion model.
    """
    try:
        # Load images
        main_image = Image.open(main_image_path).convert('RGBA')
        object_image = Image.open(object_crop_path).convert('RGBA')
        
        # Create a copy of the main image
        result_image = main_image.copy()
        
        # Resize object to fit the target box
        resized_object = resize_object_to_fit_box(object_image, target_box)
        
        # Calculate paste position (center the object in the box)
        x1, y1, x2, y2 = target_box
        box_width = x2 - x1
        box_height = y2 - y1
        obj_width, obj_height = resized_object.size
        
        paste_x = x1 + (box_width - obj_width) // 2
        paste_y = y1 + (box_height - obj_height) // 2
        
        # Paste the object
        result_image.paste(resized_object, (paste_x, paste_y), resized_object)
        
        # Save result
        if output_path is None:
            output_path = "output/simple_paste_result.jpg"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert back to RGB for JPEG saving
        result_image_rgb = result_image.convert('RGB')
        result_image_rgb.save(output_path, quality=95)
        
        return InsertionResult(
            success=True,
            output_path=output_path,
            bounding_box=target_box,
            confidence_score=0.9  # High confidence for simple pasting
        )
        
    except Exception as e:
        return InsertionResult(
            success=False,
            output_path="",
            error_message=f"Simple pasting failed: {str(e)}"
        )

def simple_text_paste(
    main_image_path: str,
    text: str,
    target_box: Tuple[int, int, int, int],
    font_size: int = 48,
    font_color: str = "white",
    background_color: str = None,
    output_path: str = None,
) -> InsertionResult:
    """
    Simple text pasting without diffusion model.
    Places text directly in the target bounding box with transparent background.
    """
    try:
        # Load main image
        main_image = Image.open(main_image_path).convert('RGBA')
        
        # Create text image that fits the target box with transparent background
        text_image = create_text_image_for_box(text, target_box, font_size, font_color)
        
        # Create a copy of the main image
        result_image = main_image.copy()
        
        # Paste the text at the target box position
        x1, y1, x2, y2 = target_box
        result_image.paste(text_image, (x1, y1), text_image)
        
        # Save result
        if output_path is None:
            output_path = f"output/simple_text_{text.replace(' ', '_')}.jpg"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert back to RGB for JPEG saving
        result_image_rgb = result_image.convert('RGB')
        result_image_rgb.save(output_path, quality=95)
        
        return InsertionResult(
            success=True,
            output_path=output_path,
            bounding_box=target_box,
            confidence_score=0.9  # High confidence for simple pasting
        )
        
    except Exception as e:
        return InsertionResult(
            success=False,
            output_path="",
            error_message=f"Simple text pasting failed: {str(e)}"
        ) 