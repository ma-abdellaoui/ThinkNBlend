import os
import json
import subprocess
from think_n_blend.schemas import BoundingBox
from think_n_blend.utils.image_utils import create_mask_from_box
from think_n_blend.services.model_manager import model_manager
from think_n_blend.services.simple_paste_service import simple_object_paste

def blend_object_with_unicombine(
    main_image_path: str,
    object_crop_path: str,
    inpainting_description: str,
    target_box: BoundingBox,
    diffusion_model: str = "unicombine",
    output_dir: str = "output",
) -> str | None:
    """
    Blends the object into the scene using the specified diffusion model.
    """
    print(f"\n--- Running {diffusion_model} Blending ---")

    # Check if using simple paste model
    if model_manager.is_simple_paste_model(diffusion_model):
        print("Using simple paste mode (no diffusion model required)")
        result = simple_object_paste(
            main_image_path,
            object_crop_path,
            target_box,
            os.path.join(output_dir, "simple_paste_result.jpg")
        )
        
        if result.success:
            return result.output_path
        else:
            print(f"Simple paste failed: {result.error_message}")
            return None

    # Check if model is available
    if not model_manager.check_model_availability(diffusion_model, "diffusion"):
        print(f"Error: {diffusion_model} model not available")
        return None

    mask_path = create_mask_from_box(main_image_path, target_box, os.path.join(output_dir, "mask.png"))
    
    unicombine_json_data = {
        "bg_prompt": "background",
        "fg_prompt": inpainting_description,
        "box": [int(v) for v in target_box],
        "fg_keep_original": False,
    }
    unicombine_json_path = os.path.join(output_dir, "unicombine_data.json")
    with open(unicombine_json_path, 'w') as f:
        json.dump(unicombine_json_data, f)

    # Get inference command from model manager
    command = model_manager.get_inference_command(
        diffusion_model,
        main_image_path=main_image_path,
        object_crop_path=object_crop_path,
        json_path=unicombine_json_path,
        output_dir=output_dir
    )

    try:
        subprocess.run(command, check=True)
        
        output_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png')) and "mask" not in f and "visualization" not in f]
        if not output_files:
            print("Error: No output image found from diffusion model.")
            return None
            
        latest_file = max(output_files, key=os.path.getctime)
        final_image_path = os.path.join(output_dir, "final_blended_image.jpg")
        os.rename(latest_file, final_image_path)
        return final_image_path

    except subprocess.CalledProcessError as e:
        print(f"Error during {diffusion_model} execution: {e}")
        return None
