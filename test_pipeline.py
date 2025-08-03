#!/usr/bin/env python3
"""
Test script for ThinkNBlend pipeline.
Demonstrates object and text insertion capabilities using sample images.
"""

import os
import sys
import glob
from PIL import Image, ImageDraw, ImageFont
from think_n_blend.cli import object_insertion_pipeline, text_insertion_pipeline

def find_sample_images():
    """Find all sample images and their corresponding objects."""
    sample_inputs_dir = "sample_inputs"
    sample_outputs_dir = "sample_outputs"
    
    # Create output directory if it doesn't exist
    os.makedirs(sample_outputs_dir, exist_ok=True)
    
    # Find all main sample images (sample_n.jpg/png)
    main_images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        main_images.extend(glob.glob(os.path.join(sample_inputs_dir, ext)))
    
    # Filter to only include files that match sample_n pattern
    main_images = [img for img in main_images if os.path.basename(img).startswith('sample_') and not '_obj_' in img]
    
    # Find corresponding object images for each main image
    sample_data = []
    for main_img in main_images:
        base_name = os.path.splitext(os.path.basename(main_img))[0]  # e.g., "sample_1"
        obj_images = []
        
        # Find all object images for this sample
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            obj_pattern = os.path.join(sample_inputs_dir, f"{base_name}_obj_*{ext[1:]}")
            obj_images.extend(glob.glob(obj_pattern))
        
        if obj_images:
            sample_data.append({
                'main_image': main_img,
                'object_images': obj_images,
                'sample_name': base_name
            })
    
    return sample_data, sample_outputs_dir

def test_object_insertion_with_samples():
    """Test object insertion using sample images."""
    print("\n" + "="*50)
    print("TESTING OBJECT INSERTION WITH SAMPLE IMAGES")
    print("="*50)
    
    sample_data, output_dir = find_sample_images()
    
    if not sample_data:
        print("❌ No sample images found in sample_inputs directory")
        print("Expected format: sample_1.jpg, sample_1_obj_1.jpg, sample_1_obj_2.jpg, etc.")
        return
    
    print(f"Found {len(sample_data)} sample images with objects:")
    for sample in sample_data:
        print(f"  - {sample['main_image']} with {len(sample['object_images'])} objects")
    
    results = []
    for sample in sample_data:
        print(f"\n--- Processing {sample['sample_name']} ---")
        
        for i, obj_img in enumerate(sample['object_images']):
            print(f"  Inserting object {i+1}/{len(sample['object_images'])}: {os.path.basename(obj_img)}")
            
            try:
                result_path = object_insertion_pipeline(
                    sample['main_image'],
                    obj_img,
                    verify=True
                )
                
                if result_path:
                    # Move result to sample_outputs with descriptive name
                    output_filename = f"{sample['sample_name']}_obj_{i+1}_result.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Copy the result to sample_outputs
                    import shutil
                    shutil.copy2(result_path, output_path)
                    
                    results.append({
                        'sample': sample['sample_name'],
                        'object': os.path.basename(obj_img),
                        'output': output_path,
                        'success': True
                    })
                    print(f"    ✅ Success: {output_filename}")
                else:
                    results.append({
                        'sample': sample['sample_name'],
                        'object': os.path.basename(obj_img),
                        'output': None,
                        'success': False,
                        'error': 'Pipeline failed'
                    })
                    print(f"    ❌ Failed: {os.path.basename(obj_img)}")
                    
            except Exception as e:
                results.append({
                    'sample': sample['sample_name'],
                    'object': os.path.basename(obj_img),
                    'output': None,
                    'success': False,
                    'error': str(e)
                })
                print(f"    ❌ Error: {str(e)}")
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    print(f"\n" + "="*50)
    print(f"OBJECT INSERTION SUMMARY: {successful}/{total} successful")
    print("="*50)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['sample']} - {result['object']}")
        if not result['success'] and 'error' in result:
            print(f"    Error: {result['error']}")
    
    return results

def test_text_insertion_with_samples():
    """Test text insertion using sample images."""
    print("\n" + "="*50)
    print("TESTING TEXT INSERTION WITH SAMPLE IMAGES")
    print("="*50)
    
    sample_data, output_dir = find_sample_images()
    
    if not sample_data:
        print("❌ No sample images found in sample_inputs directory")
        return
    
    # Sample texts to insert
    sample_texts = [ "BRAND", "LOGO", "DEMO"]
    positions = ["top", "bottom"]
    
    results = []
    for sample in sample_data:
        print(f"\n--- Processing {sample['sample_name']} ---")
        
        for text in sample_texts:
            for position in positions:
                print(f"  Inserting text '{text}' at {position}")
                
                try:
                    result_path = text_insertion_pipeline(
                        sample['main_image'],
                        text,
                        position,
                        verify=True
                    )
                    
                    if result_path:
                        # Move result to sample_outputs with descriptive name
                        output_filename = f"{sample['sample_name']}_text_{text}_{position}_result.jpg"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Copy the result to sample_outputs
                        import shutil
                        shutil.copy2(result_path, output_path)
                        
                        results.append({
                            'sample': sample['sample_name'],
                            'text': text,
                            'position': position,
                            'output': output_path,
                            'success': True
                        })
                        print(f"    ✅ Success: {output_filename}")
                    else:
                        results.append({
                            'sample': sample['sample_name'],
                            'text': text,
                            'position': position,
                            'output': None,
                            'success': False,
                            'error': 'Pipeline failed'
                        })
                        print(f"    ❌ Failed: {text} at {position}")
                        
                except Exception as e:
                    results.append({
                        'sample': sample['sample_name'],
                        'text': text,
                        'position': position,
                        'output': None,
                        'success': False,
                        'error': str(e)
                    })
                    print(f"    ❌ Error: {str(e)}")
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    print(f"\n" + "="*50)
    print(f"TEXT INSERTION SUMMARY: {successful}/{total} successful")
    print("="*50)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['sample']} - '{result['text']}' at {result['position']}")
        if not result['success'] and 'error' in result:
            print(f"    Error: {result['error']}")
    
    return results

def main():
    """Run the test pipeline with sample images."""
    print("ThinkNBlend Pipeline Test with Sample Images")
    print("="*50)
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
    
    # Check UniCombine setup
    unicombine_path = "submodules/UniCombine/inference.py"
    if not os.path.exists(unicombine_path):
        print("❌ Error: UniCombine not found. Please set up the submodules:")
        print("   git submodule update --init --recursive")
        print("   cd submodules/UniCombine")
        print("   # Follow UniCombine README for setup instructions")
        return
    
    # Check if UniCombine requirements are installed
    try:
        import torch
        print("✅ PyTorch is available")
    except ImportError:
        print("❌ Error: PyTorch not found. Please install requirements:")
        print("   pip install -r requirements.txt")
        return
    
    # Check sample inputs directory
    if not os.path.exists("sample_inputs"):
        print("❌ Error: sample_inputs directory not found")
        print("   Please create sample_inputs directory with sample images")
        return
    
    # Test object insertion with sample images
    object_results = test_object_insertion_with_samples()
    
    # Test text insertion with sample images
    text_results = test_text_insertion_with_samples()
    
    print("\n" + "="*50)
    print("TEST COMPLETE")
    print("="*50)
    print("Check the 'sample_outputs' directory for results.")
    print("Sample images should be in 'sample_inputs' directory.")
    print("Expected format:")
    print("  sample_inputs/")
    print("    sample_1.jpg")
    print("    sample_1_obj_1.jpg")
    print("    sample_1_obj_2.jpg")
    print("  sample_outputs/")
    print("    sample_1_obj_1_result.jpg")
    print("    sample_1_obj_2_result.jpg")
    print("    sample_1_text_BRAND_top_result.jpg")

if __name__ == "__main__":
    main() 