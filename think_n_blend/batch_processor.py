import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from think_n_blend.cli import object_insertion_pipeline, text_insertion_pipeline
from think_n_blend.schemas import InsertionResult

class BatchProcessor:
    """Handles batch processing of multiple images for object and text insertion."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_object_insertions(self, object_crops_dir: str, verify: bool = False) -> List[Dict[str, Any]]:
        """Process object insertions for multiple images."""
        results = []
        object_crops_dir = Path(object_crops_dir)
        
        # Get all main images
        main_images = list(self.input_dir.glob("*.jpg")) + list(self.input_dir.glob("*.png"))
        object_crops = list(object_crops_dir.glob("*.jpg")) + list(object_crops_dir.glob("*.png"))
        
        print(f"Found {len(main_images)} main images and {len(object_crops)} object crops")
        
        for i, main_image in enumerate(main_images):
            for j, object_crop in enumerate(object_crops):
                print(f"\nProcessing {i+1}/{len(main_images)} main image with {j+1}/{len(object_crops)} object crop")
                
                try:
                    result_path = object_insertion_pipeline(
                        str(main_image),
                        str(object_crop),
                        verify
                    )
                    
                    if result_path:
                        results.append({
                            'main_image': str(main_image),
                            'object_crop': str(object_crop),
                            'output_path': result_path,
                            'success': True
                        })
                    else:
                        results.append({
                            'main_image': str(main_image),
                            'object_crop': str(object_crop),
                            'success': False,
                            'error': 'Pipeline failed'
                        })
                        
                except Exception as e:
                    results.append({
                        'main_image': str(main_image),
                        'object_crop': str(object_crop),
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def process_text_insertions(self, texts: List[str], positions: List[str] = None, verify: bool = False) -> List[Dict[str, Any]]:
        """Process text insertions for multiple images."""
        results = []
        
        if positions is None:
            positions = ["top", "bottom", "left", "right"]
        
        # Get all main images
        main_images = list(self.input_dir.glob("*.jpg")) + list(self.input_dir.glob("*.png"))
        
        print(f"Found {len(main_images)} main images")
        
        for i, main_image in enumerate(main_images):
            for j, text in enumerate(texts):
                for k, position in enumerate(positions):
                    print(f"\nProcessing {i+1}/{len(main_images)} main image with text '{text}' at position {position}")
                    
                    try:
                        result_path = text_insertion_pipeline(
                            str(main_image),
                            text,
                            position,
                            verify
                        )
                        
                        if result_path:
                            results.append({
                                'main_image': str(main_image),
                                'text': text,
                                'position': position,
                                'output_path': result_path,
                                'success': True
                            })
                        else:
                            results.append({
                                'main_image': str(main_image),
                                'text': text,
                                'position': position,
                                'success': False,
                                'error': 'Pipeline failed'
                            })
                            
                    except Exception as e:
                        results.append({
                            'main_image': str(main_image),
                            'text': text,
                            'position': position,
                            'success': False,
                            'error': str(e)
                        })
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save processing results to JSON file."""
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
        
        # Print summary
        successful = sum(1 for r in results if r.get('success', False))
        total = len(results)
        print(f"Processing complete: {successful}/{total} successful insertions")

def main():
    parser = argparse.ArgumentParser(description="Batch processing for ThinkNBlend")
    parser.add_argument("--mode", choices=["object", "text"], required=True,
                       help="Insertion mode")
    parser.add_argument("--input_dir", type=str, default="input",
                       help="Directory containing main images")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Directory for output images")
    parser.add_argument("--object_crops_dir", type=str,
                       help="Directory containing object crops (for object mode)")
    parser.add_argument("--texts", type=str, nargs="+",
                       help="List of texts to insert (for text mode)")
    parser.add_argument("--positions", type=str, nargs="+", 
                       default=["top", "bottom", "left", "right"],
                       help="Text positions (for text mode)")
    parser.add_argument("--verify", action="store_true",
                       help="Enable verification for all insertions")
    parser.add_argument("--output_file", type=str, default="batch_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    processor = BatchProcessor(args.input_dir, args.output_dir)
    
    if args.mode == "object":
        if not args.object_crops_dir:
            parser.error("--object_crops_dir is required for object mode")
        
        results = processor.process_object_insertions(args.object_crops_dir, args.verify)
        
    elif args.mode == "text":
        if not args.texts:
            parser.error("--texts is required for text mode")
        
        results = processor.process_text_insertions(args.texts, args.positions, args.verify)
    
    processor.save_results(results, args.output_file)

if __name__ == "__main__":
    main() 