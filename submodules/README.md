# Submodules

This directory contains external model repositories managed as git submodules. This modular approach allows ThinkNBlend to support multiple diffusion and object detection models while keeping the main repository clean.

## Structure

```
submodules/
├── README.md           # This file
├── UniCombine/         # UniCombine diffusion model (git submodule)
└── [future models]     # Additional models can be added here
```

## Current Models

### UniCombine

- **Repository**: https://github.com/Xuan-World/UniCombine
- **Purpose**: Object and text insertion using Stable Diffusion
- **Type**: Diffusion Model
- **Status**: Active
- **Requirements**: 34GB+ VRAM, Python 3.10+

**Setup Instructions**:

1. **Navigate to UniCombine directory**:

   ```bash
   cd submodules/UniCombine
   ```

2. **Follow UniCombine README instructions**:

   - Download the required model weights
   - Install UniCombine-specific requirements
   - Set up the model checkpoints

3. **Install UniCombine requirements**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights** (refer to UniCombine documentation for specific URLs):

   - FLUX.1-schnell model
   - Condition-LoRA weights
   - Denoising-LoRA weights

5. **Verify installation**:
   ```bash
   python inference.py --help
   ```

**Note**: The exact download instructions and model URLs can be found in the UniCombine repository README at `submodules/UniCombine/README.md`.

## Adding New Models

To add a new model to the system:

1. **Add as Git Submodule**:

   ```bash
   git submodule add <repository-url> submodules/<model-name>
   ```

2. **Update Configuration**:
   Edit `think_n_blend/config.py` to add the new model configuration:

   ```python
   DIFFUSION_MODELS = {
       "unicombine": { ... },
       "new_model": {
           "path": f"{SUBMODULES_DIR}/new_model",
           "inference_script": "inference.py",
           "requirements": "requirements.txt",
           "description": "Description of the new model"
       }
   }
   ```

3. **Implement Model Interface**:
   Add the inference command logic in `think_n_blend/services/model_manager.py`:

   ```python
   def _get_new_model_command(self, inference_script: str, **kwargs) -> list:
       # Return the command for the new model
       return [...]
   ```

4. **Update Model Manager**:
   Add the new model to the `get_inference_command` method.

## Managing Submodules

### Initial Setup

```bash
# Clone the repository with submodules
git clone --recursive <repository-url>

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### Updating Submodules

```bash
# Update all submodules to latest versions
git submodule update --remote

# Update specific submodule
git submodule update --remote submodules/UniCombine
```

### Working with Submodules

```bash
# Enter submodule directory
cd submodules/UniCombine

# Make changes and commit
git add .
git commit -m "Update model"

# Update main repository
cd ../..
git add submodules/UniCombine
git commit -m "Update UniCombine submodule"
```

## Model Requirements

Each model in the submodules directory should have:

1. **Inference Script**: Main script for running the model
2. **Requirements**: Python dependencies
3. **Documentation**: README explaining usage
4. **Configuration**: Model-specific settings

## Best Practices

1. **Version Pinning**: Use specific commits for production stability
2. **Documentation**: Keep model-specific docs in the submodule
3. **Testing**: Test new models before adding to main pipeline
4. **Compatibility**: Ensure models work with the existing pipeline interface

## Troubleshooting

### Submodule Issues

```bash
# Reset submodule to tracked state
git submodule update --init --recursive

# Remove and re-add submodule
git submodule deinit submodules/UniCombine
git rm submodules/UniCombine
git submodule add <repository-url> submodules/UniCombine
```

### Model Loading Issues

- Check if the model path exists
- Verify inference script is executable
- Ensure all dependencies are installed
- Check model configuration in `config.py`

## Future Models

Potential models to add:

1. **Stable Diffusion**: For general image generation
2. **ControlNet**: For controlled image generation
3. **DALL-E**: For text-to-image generation
4. **Custom Models**: Domain-specific models

## Contributing

When adding new models:

1. Follow the existing model interface
2. Add comprehensive documentation
3. Include example usage
4. Test thoroughly before submission
5. Update the main README with new capabilities
