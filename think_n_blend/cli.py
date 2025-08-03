import argparse
import os
from think_n_blend.services import (
    vision_service, detection_service, composition_service, 
    blending_service, text_service, verification_service
)
from think_n_blend.schemas import TextInsertion
from think_n_blend.utils.image_utils import create_dummy_image, save_bounding_box_visualization
from think_n_blend.services.model_manager import model_manager

def object_insertion_pipeline(main_image: str, object_crop: str, verify: bool = False, diffusion_model: str = "unicombine"):
    """Runs the object insertion pipeline."""
    print("=== Object Insertion Pipeline ===")
    
    # Check model availability
    if not model_manager.check_model_availability(diffusion_model, "diffusion"):
        print(f"Error: {diffusion_model} model not available")
        return None
    
    # --- Stage 1: GPT-4 Vision Reasoning ---
    print("\n--- Stage 1: GPT-4 Vision Reasoning ---")
    try:
        vision_response = vision_service.get_vision_reasoning(main_image, object_crop)
        print(f"Reference Object Label: {vision_response.reference_object.label}")
        print(f"Relative Position: {vision_response.target_object.relative_position}")
        print(f"Inpainting Description: {vision_response.target_object.inpainting_description}")
    except Exception as e:
        print(f"Error in Stage 1: {e}")
        return None
    print("---------------------------------------------")

    # --- Stage 2: Zero-Shot Object Detection ---
    print("\n--- Stage 2: Zero-Shot Object Detection ---")
    reference_box = detection_service.detect_reference_object(
        main_image, vision_response.reference_object.label
    )
    if not reference_box:
        print(f"Could not detect '{vision_response.reference_object.label}' in the image.")
        return None
    print(f"Detected reference box: {reference_box}")
    print("-----------------------------------------")

    # --- Stage 3: Compute Target Insertion Bounding Box ---
    print("\n--- Stage 3: Compute Target Bounding Box ---")
    target_box = composition_service.compute_target_bounding_box(
        main_image, reference_box, vision_response.target_object.relative_position
    )
    print(f"Computed target box: {target_box}")
    print("------------------------------------------")

    # --- Stage 4: Stable Diffusion Blending ---
    final_image_path = blending_service.blend_object_with_unicombine(
        main_image,
        object_crop,
        vision_response.target_object.inpainting_description,
        target_box,
        diffusion_model
    )

    if final_image_path:
        print(f"\nPipeline complete. Final image saved at: {final_image_path}")
        
        # Verification
        if verify:
            print("\n--- Verification ---")
            verification_result = verification_service.verify_insertion_quality(
                final_image_path, "object", vision_response.target_object.label
            )
            print(f"Object detected: {verification_result.object_detected}")
            print(f"Confidence: {verification_result.object_confidence}")
        
        save_bounding_box_visualization(
            main_image,
            reference_box,
            target_box,
            "output/object_bounding_boxes_visualization.jpg",
        )
        print("Saved visualization with reference and target boxes")
        return final_image_path
    else:
        print("\nPipeline failed at the blending stage.")
        return None

def text_insertion_pipeline(main_image: str, text: str, position: str = "top", verify: bool = False, diffusion_model: str = "unicombine"):
    """Runs the text insertion pipeline."""
    print("=== Text Insertion Pipeline ===")
    
    # Check model availability
    if not model_manager.check_model_availability(diffusion_model, "diffusion"):
        print(f"Error: {diffusion_model} model not available")
        return None
    
    # Create text insertion object
    text_insertion = TextInsertion(
        text=text,
        position=position,
        inpainting_description=f"text saying '{text}' in white color"
    )
    
    # Compute target box based on position
    from PIL import Image
    img = Image.open(main_image)
    img_width, img_height = img.size
    
    if position == "top":
        target_box = (0, 0, img_width, img_height // 4)
    elif position == "bottom":
        target_box = (0, 3 * img_height // 4, img_width, img_height)
    elif position == "left":
        target_box = (0, 0, img_width // 4, img_height)
    elif position == "right":
        target_box = (3 * img_width // 4, 0, img_width, img_height)
    else:
        target_box = (0, 0, img_width, img_height // 4)
    
    # Insert text
    result = text_service.insert_text_with_unicombine(
        main_image,
        text_insertion,
        target_box,
        diffusion_model
    )
    
    if result.success:
        print(f"\nText insertion complete. Final image saved at: {result.output_path}")
        
        # Verification
        if verify:
            print("\n--- Verification ---")
            verification_result = verification_service.verify_insertion_quality(
                result.output_path, "text", text
            )
            print(f"Text detected: {verification_result.text_detected}")
            print(f"Detected text: {verification_result.detected_text}")
            print(f"Confidence: {verification_result.text_confidence}")
        
        return result.output_path
    else:
        print(f"\nText insertion failed: {result.error_message}")
        return None

def list_models():
    """List available models."""
    models = model_manager.list_available_models()
    print("Available Models:")
    print("=================")
    print("Diffusion Models:")
    for model in models["diffusion_models"]:
        config = model_manager.get_diffusion_model_config(model)
        print(f"  - {model}: {config['description']}")
    print("\nObject Detection Models:")
    for model in models["object_detection_models"]:
        config = model_manager.get_object_detection_model_config(model)
        print(f"  - {model}: {config['description']}")

def main():
    parser = argparse.ArgumentParser(description="ThinkNBlend: Context-aware object and text insertion pipeline.")
    parser.add_argument("--mode", choices=["object", "text", "list-models"], required=True, 
                       help="Insertion mode: object, text, or list-models")
    parser.add_argument("--main_image", type=str, default="input/main_image.jpg", 
                       help="Path to the main image.")
    parser.add_argument("--object_crop", type=str, default="input/object_crop.jpg", 
                       help="Path to the object crop image (for object mode).")
    parser.add_argument("--text", type=str, help="Text to insert (for text mode).")
    parser.add_argument("--position", type=str, default="top", 
                       choices=["top", "bottom", "left", "right"],
                       help="Position for text insertion (for text mode).")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify insertion quality using object detection/OCR.")
    parser.add_argument("--diffusion_model", type=str, default="unicombine",
                       help="Diffusion model to use for blending.")
    
    args = parser.parse_args()

    if args.mode == "list-models":
        list_models()
        return

    # Input validation
    if args.mode == "object":
        if not os.path.exists(args.main_image):
            print(f"Main image not found at '{args.main_image}'. Creating a dummy file.")
            create_dummy_image(args.main_image, (800, 600), 'red')
        if not os.path.exists(args.object_crop):
            print(f"Object crop not found at '{args.object_crop}'. Creating a dummy file.")
            create_dummy_image(args.object_crop, (100, 100), 'blue')
        
        return object_insertion_pipeline(args.main_image, args.object_crop, args.verify, args.diffusion_model)
    
    elif args.mode == "text":
        if not args.text:
            parser.error("--text is required for text mode")
        
        if not os.path.exists(args.main_image):
            print(f"Main image not found at '{args.main_image}'. Creating a dummy file.")
            create_dummy_image(args.main_image, (800, 600), 'red')
        
        return text_insertion_pipeline(args.main_image, args.text, args.position, args.verify, args.diffusion_model)

if __name__ == "__main__":
    main()
