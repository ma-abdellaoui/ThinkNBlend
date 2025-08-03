import os
import subprocess
from typing import Dict, Any, Optional
from think_n_blend.config import DIFFUSION_MODELS, OBJECT_DETECTION_MODELS, DEFAULT_DIFFUSION_MODEL, DEFAULT_OBJECT_DETECTION_MODEL

class ModelManager:
    """Manages different diffusion and object detection models."""
    
    def __init__(self):
        self.diffusion_models = DIFFUSION_MODELS
        self.object_detection_models = OBJECT_DETECTION_MODELS
        self.current_diffusion_model = DEFAULT_DIFFUSION_MODEL
        self.current_object_detection_model = DEFAULT_OBJECT_DETECTION_MODEL
    
    def get_diffusion_model_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a diffusion model."""
        model_name = model_name or self.current_diffusion_model
        if model_name not in self.diffusion_models:
            raise ValueError(f"Unknown diffusion model: {model_name}")
        return self.diffusion_models[model_name]
    
    def get_object_detection_model_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for an object detection model."""
        model_name = model_name or self.current_object_detection_model
        if model_name not in self.object_detection_models:
            raise ValueError(f"Unknown object detection model: {model_name}")
        return self.object_detection_models[model_name]
    
    def list_available_models(self) -> Dict[str, list]:
        """List all available models."""
        return {
            "diffusion_models": list(self.diffusion_models.keys()),
            "object_detection_models": list(self.object_detection_models.keys())
        }
    
    def set_diffusion_model(self, model_name: str):
        """Set the current diffusion model."""
        if model_name not in self.diffusion_models:
            raise ValueError(f"Unknown diffusion model: {model_name}")
        self.current_diffusion_model = model_name
    
    def set_object_detection_model(self, model_name: str):
        """Set the current object detection model."""
        if model_name not in self.object_detection_models:
            raise ValueError(f"Unknown object detection model: {model_name}")
        self.current_object_detection_model = model_name
    
    def check_model_availability(self, model_name: str, model_type: str = "diffusion") -> bool:
        """Check if a model is available and properly installed."""
        if model_type == "diffusion":
            config = self.get_diffusion_model_config(model_name)
            model_path = config["path"]
            inference_script = os.path.join(model_path, config["inference_script"])
            return os.path.exists(inference_script)
        elif model_type == "object_detection":
            # For Hugging Face models, we assume they're available if the config exists
            return model_name in self.object_detection_models
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def install_model_requirements(self, model_name: str) -> bool:
        """Install requirements for a specific model."""
        try:
            config = self.get_diffusion_model_config(model_name)
            requirements_path = os.path.join(config["path"], config["requirements"])
            
            if os.path.exists(requirements_path):
                subprocess.run([
                    "pip", "install", "-r", requirements_path
                ], check=True)
                return True
            else:
                print(f"Warning: No requirements.txt found for {model_name}")
                return False
        except Exception as e:
            print(f"Error installing requirements for {model_name}: {e}")
            return False
    
    def get_inference_command(self, model_name: str, **kwargs) -> list:
        """Get the inference command for a diffusion model."""
        config = self.get_diffusion_model_config(model_name)
        inference_script = os.path.join(config["path"], config["inference_script"])
        
        if model_name == "unicombine":
            return self._get_unicombine_command(inference_script, **kwargs)
        else:
            raise ValueError(f"No inference command defined for model: {model_name}")
    
    def _get_unicombine_command(self, inference_script: str, **kwargs) -> list:
        """Get UniCombine inference command."""
        command = [
            "python", inference_script,
            "--condition_types", "fill", "subject",
            "--denoising_lora_name", "subject_fill_union",
            "--denoising_lora_weight", "1.0",
            "--fill", kwargs.get("main_image_path", ""),
            "--subject", kwargs.get("object_crop_path", ""),
            "--json", kwargs.get("json_path", ""),
            "--version", "training-based",
            "--output_dir", kwargs.get("output_dir", "output"),
        ]
        return command

# Global model manager instance
model_manager = ModelManager() 