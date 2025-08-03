# ThinkNBlend: Context-Aware Object and Text Insertion Model

## Model Summary

ThinkNBlend is a multi-stage pipeline that performs context-aware insertion of synthetic objects and text into real-world images. The system combines GPT-4 Vision reasoning, zero-shot object detection, and Stable Diffusion-based blending to create realistic composite images.

### Method Summary

The pipeline operates in four main stages:

1. **GPT-4 Vision Reasoning**: Analyzes the main image and object/text to determine optimal placement
2. **Zero-Shot Object Detection**: Uses OWLv2 model to detect reference objects in the scene
3. **Bounding Box Computation**: Calculates target insertion area based on relative positioning
4. **Stable Diffusion Blending**: Uses UniCombine to seamlessly blend objects/text into the scene

### Key Features

- **Object Insertion**: Realistically inserts objects (e.g., hats, bottles) into scenes
- **Text Insertion**: Adds text with customizable fonts, colors, and positions
- **Context Awareness**: Uses AI reasoning to determine natural placement
- **Quality Verification**: Optional object detection and OCR verification
- **Docker Deployment**: Containerized for easy GPU deployment

## Inputs / Outputs

### Supported Input Formats

**Object Insertion:**

- Main image: JPEG, PNG (any resolution)
- Object crop: JPEG, PNG (cropped object image)
- Output: JPEG with inserted object

**Text Insertion:**

- Main image: JPEG, PNG (any resolution)
- Text: String with font properties
- Position: top, bottom, left, right
- Output: JPEG with inserted text

### Output Specifications

- **Format**: JPEG
- **Quality**: High-resolution with realistic blending
- **Metadata**: Includes bounding box coordinates and confidence scores
- **Verification**: Optional quality assessment reports

## Environment

### Dependencies

**Core Requirements:**

- Python 3.10+
- PyTorch 2.4.1
- Transformers 4.46.3
- OpenAI API access
- CUDA 11.8+ (for GPU acceleration)

**Additional Libraries:**

- EasyOCR (for text verification)
- Pillow (image processing)
- OpenCV (computer vision)

### Hardware Requirements

**Minimum:**

- 8GB RAM
- 34GB VRAM (NVIDIA GPU)
- 10GB storage

**Recommended:**

- 16GB RAM
- 40GB+ VRAM (NVIDIA GPU)
- 20GB storage

### Performance Metrics

- **Processing Time**: 30-60 seconds per image (GPU)
- **Memory Usage**: 34-40GB VRAM during processing
- **Batch Processing**: Supported for multiple images
- **Scalability**: Linear scaling with GPU memory

## Setup Instructions

### UniCombine Model Setup

**IMPORTANT**: Before using ThinkNBlend, you must follow the UniCombine repository instructions to download the required weights and models:

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

## Setup Instructions

### UniCombine Model Setup

**IMPORTANT**: Before using ThinkNBlend, you must follow the UniCombine repository instructions to download the required weights and models:

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

## Limitations

### Technical Limitations

1. **Object Size**: Very small or very large objects may not blend well
2. **Complex Scenes**: Highly cluttered backgrounds may affect placement accuracy
3. **Text Length**: Very long text strings may not fit properly
4. **Lighting**: Extreme lighting conditions may affect realism
5. **GPU Dependency**: Requires NVIDIA GPU with 34GB+ VRAM for optimal performance

### Quality Limitations

1. **Artifacts**: Occasional blending artifacts at object boundaries
2. **Perspective**: May not perfectly match scene perspective
3. **Shadows**: Limited shadow generation for inserted objects
4. **Reflections**: No automatic reflection generation
5. **Occlusion**: Limited handling of object occlusion

### Edge Cases

1. **Transparent Objects**: May not handle transparency well
2. **Moving Objects**: Designed for static scenes
3. **Multiple Objects**: Sequential insertion only
4. **Text Orientation**: Limited to horizontal text
5. **Language Support**: Primarily English text

## Improvement Ideas

### Short-term Improvements

1. **Enhanced Blending**: Implement advanced blending algorithms
2. **Shadow Generation**: Add automatic shadow casting
3. **Perspective Matching**: Improve perspective alignment
4. **Batch Processing**: Optimize for multiple insertions
5. **Error Handling**: Better error recovery and reporting

### Long-term Enhancements

1. **Multi-object Insertion**: Support simultaneous multiple objects
2. **Dynamic Lighting**: Automatic lighting adjustment
3. **3D Integration**: Support for 3D object models
4. **Video Support**: Extend to video sequences
5. **Custom Training**: Fine-tune models for specific domains

### Research Directions

1. **Neural Rendering**: Implement neural rendering techniques
2. **Physics Integration**: Add physics-based constraints
3. **Style Transfer**: Maintain consistent artistic styles
4. **Interactive Editing**: Real-time editing capabilities
5. **Quality Metrics**: Automated quality assessment

## Reusability

### Domain Adaptability

**Current Domains:**

- Photography enhancement
- E-commerce product placement
- Marketing material creation
- Educational content generation

**Potential Extensions:**

- Medical imaging annotation
- Architectural visualization
- Gaming asset generation
- Scientific illustration
- Legal evidence enhancement

### Customization Options

1. **Model Fine-tuning**: Adapt to specific object categories
2. **Style Transfer**: Match different artistic styles
3. **Domain-specific Prompts**: Customize reasoning prompts
4. **Quality Thresholds**: Adjust verification sensitivity
5. **Output Formats**: Support various output specifications

### Integration Capabilities

1. **API Interface**: RESTful API for web integration
2. **Plugin Architecture**: Modular service design
3. **Cloud Deployment**: Docker containerization
4. **Batch Processing**: Command-line interface
5. **Quality Monitoring**: Automated verification pipeline

## Usage Examples

### Object Insertion

```bash
python main.py --mode object \
  --main_image input/scene.jpg \
  --object_crop input/hat.png \
  --verify
```

### Text Insertion

```bash
python main.py --mode text \
  --main_image input/scene.jpg \
  --text "BRAND" \
  --position top \
  --verify
```

### Docker Deployment

```bash
docker-compose up --build
```

## Citation

If you use ThinkNBlend in your research, please cite:

```bibtex
@misc{thinknblend2024,
  title={ThinkNBlend: Context-Aware Object and Text Insertion Pipeline},
  author={ThinkNBlend Team},
  year={2024},
  url={https://github.com/your-repo/ThinkNBlend}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
