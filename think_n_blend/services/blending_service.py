import os
import json
import subprocess
from think_n_blend.schemas import BoundingBox
from think_n_blend.utils.image_utils import create_mask_from_box
from think_n_blend.services.model_manager import model_manager

def blend_object_with_unicombine(
    main_image_path: str,
    object_crop_path: str,
    inpainting_description: str,
    target_box: BoundingBox,
    diffusion_model: str = "unicombine",
) -> str | None:
    """
    Blends the object into the scene using the specified diffusion model.
    """
    print(f"\n--- Running {diffusion_model} Blending ---")

    # Check if model is available
    if not model_manager.check_model_availability(diffusion_model, "diffusion"):
        print(f"Error: {diffusion_model} model not available")
        return None

    mask_path = create_mask_from_box(main_image_path, target_box, "output/mask.png")
    
    unicombine_json_data = {
        "bg_prompt": "background",
        "fg_prompt": inpainting_description,
        "box": [int(v) for v in target_box],
        "fg_keep_original": False,
    }
    unicombine_json_path = "output/unicombine_data.json"
    with open(unicombine_json_path, 'w') as f:
        json.dump(unicombine_json_data, f)

    # Get inference command from model manager
    command = model_manager.get_inference_command(
        diffusion_model,
        main_image_path=main_image_path,
        object_crop_path=object_crop_path,
        json_path=unicombine_json_path,
        output_dir="output"
    )

    try:
        subprocess.run(command, check=True)
        
        output_files = [os.path.join("output", f) for f in os.listdir("output") if f.endswith(('.jpg', '.png')) and "mask" not in f and "visualization" not in f]
        if not output_files:
            print("Error: No output image found from diffusion model.")
            return None
            
        latest_file = max(output_files, key=os.path.getctime)
        final_image_path = os.path.join("output", "final_blended_image.jpg")
        os.rename(latest_file, final_image_path)
        return final_image_path

    except subprocess.CalledProcessError as e:
        print(f"Error during {diffusion_model} execution: {e}")
        return None
