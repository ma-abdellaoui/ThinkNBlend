import os
import json
from PIL import Image, ImageDraw, ImageFont
from think_n_blend.schemas import TextInsertion, InsertionResult
from think_n_blend.utils.image_utils import create_mask_from_box
from think_n_blend.services.model_manager import model_manager

def create_text_image(text: str, font_size: int = 48, font_color: str = "white", 
                     background_color: str = "black", size: tuple[int, int] = (512, 128)) -> str:
    """Creates a text image for insertion."""
    image = Image.new('RGB', size, background_color)
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill=font_color, font=font)
    
    output_path = f"output/text_{text.replace(' ', '_')}.png"
    image.save(output_path)
    return output_path

def insert_text_with_unicombine(
    main_image_path: str,
    text_insertion: TextInsertion,
    target_box: tuple[int, int, int, int],
    diffusion_model: str = "unicombine",
) -> InsertionResult:
    """
    Inserts text into the scene using the specified diffusion model.
    """
    print(f"\n--- Inserting text: '{text_insertion.text}' with {diffusion_model} ---")

    # Check if model is available
    if not model_manager.check_model_availability(diffusion_model, "diffusion"):
        return InsertionResult(
            success=False,
            output_path="",
            error_message=f"{diffusion_model} model not available"
        )

    # Create text image
    text_image_path = create_text_image(
        text_insertion.text,
        text_insertion.font_size,
        text_insertion.font_color,
        text_insertion.background_color or "black"
    )
    
    # Create mask for the insertion area
    mask_path = create_mask_from_box(main_image_path, target_box, "output/text_mask.png")
    
    # Create UniCombine JSON data
    unicombine_json_data = {
        "bg_prompt": "background",
        "fg_prompt": f"text saying '{text_insertion.text}' in {text_insertion.font_color} color",
        "box": [int(v) for v in target_box],
        "fg_keep_original": False,
    }
    unicombine_json_path = "output/text_unicombine_data.json"
    with open(unicombine_json_path, 'w') as f:
        json.dump(unicombine_json_data, f)

    # Get inference command from model manager
    command = model_manager.get_inference_command(
        diffusion_model,
        main_image_path=main_image_path,
        object_crop_path=text_image_path,
        json_path=unicombine_json_path,
        output_dir="output"
    )

    try:
        import subprocess
        subprocess.run(command, check=True)
        
        output_files = [os.path.join("output", f) for f in os.listdir("output") 
                       if f.endswith(('.jpg', '.png')) and "mask" not in f and "visualization" not in f]
        if not output_files:
            return InsertionResult(
                success=False,
                output_path="",
                error_message="No output image found from diffusion model"
            )
            
        latest_file = max(output_files, key=os.path.getctime)
        final_image_path = os.path.join("output", f"text_inserted_{text_insertion.text.replace(' ', '_')}.jpg")
        os.rename(latest_file, final_image_path)
        
        return InsertionResult(
            success=True,
            output_path=final_image_path,
            bounding_box=target_box,
            confidence_score=0.8  # Placeholder confidence
        )

    except subprocess.CalledProcessError as e:
        return InsertionResult(
            success=False,
            output_path="",
            error_message=f"Error during {diffusion_model} execution: {e}"
        ) 