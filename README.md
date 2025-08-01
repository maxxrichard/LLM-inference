# GPT-2 Inference Project

This project provides a comprehensive inference script for the [GPT-2 model](https://huggingface.co/gpt2) from Hugging Face. GPT-2 is a robust and reliable language model designed for high-quality text generation tasks.

## Features

- **Easy-to-use inference script** with command-line interface
- **Interactive mode** for continuous text generation
- **Configurable generation parameters** (temperature, top-p, max length)
- **Automatic device detection** (CPU, CUDA, MPS for Apple Silicon)
- **Multiple sequence generation** support
- **Error handling** and user-friendly interface

## Installation

1. **Clone or download this project**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install torch transformers tokenizers accelerate safetensors numpy
   ```

## Usage

### Command Line Interface

#### Basic usage with a prompt:
```bash
python tiny_llm_inference.py --prompt "The future of artificial intelligence is"
```

#### With custom parameters:
```bash
python tiny_llm_inference.py --prompt "Once upon a time" --max-new-tokens 150 --temperature 0.8 --num-sequences 2
```

#### Interactive mode:
```bash
python tiny_llm_inference.py --interactive
```

#### Force specific device:
```bash
python tiny_llm_inference.py --device cpu --prompt "Hello world"
```

### Interactive Mode

Run the script in interactive mode for continuous text generation:

```bash
python tiny_llm_inference.py --interactive
```

In interactive mode, you can:
- Enter prompts continuously
- Type `settings` to modify generation parameters
- Type `quit` or `exit` to stop
- Use Ctrl+C to interrupt

### Command Line Arguments

- `--prompt`: Text prompt for generation
- `--max-new-tokens`: Maximum number of new tokens to generate (default: 80)
- `--temperature`: Sampling temperature 0.0-1.0 (default: 0.7)
- `--top-p`: Top-p sampling parameter 0.0-1.0 (default: 0.9)
- `--num-sequences`: Number of sequences to generate (default: 1)
- `--device`: Force specific device (`cpu`, `cuda`, `mps`)
- `--interactive`: Run in interactive mode

## Examples

### Example 1: Story Generation
```bash
python tiny_llm_inference.py --prompt "In a distant galaxy" --max-new-tokens 200 --temperature 0.8
```

### Example 2: Multiple Variations
```bash
python tiny_llm_inference.py --prompt "The key to success is" --num-sequences 3 --temperature 0.9
```

### Example 3: Conservative Generation
```bash
python tiny_llm_inference.py --prompt "Python programming" --temperature 0.3 --top-p 0.8
```

## Model Information

- **Model**: [GPT-2](https://huggingface.co/gpt2)
- **Type**: Causal Language Model
- **Use Cases**: Text generation, completion, creative writing, conversation
- **Size**: ~500MB (compact yet powerful)
- **Quality**: High-quality, coherent text generation

## Why GPT-2?

GPT-2 provides significantly better text generation quality compared to smaller models:
- **Coherent responses** - Generates meaningful, contextually appropriate text
- **Better grammar** - Follows proper sentence structure and language rules  
- **Creative capabilities** - Excellent for storytelling and creative writing
- **Reliable performance** - Consistent, predictable output quality
- **Well-trained** - Extensively trained on diverse, high-quality text data

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: At least 4GB RAM recommended
- **Storage**: ~500MB for model files
- **GPU**: Optional but recommended for faster inference

### Device Support

- **CPU**: Supported on all systems
- **CUDA**: For NVIDIA GPUs
- **MPS**: For Apple Silicon Macs (M1/M2/M3)

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `max-new-tokens` or use CPU instead of GPU
2. **Slow Generation**: Consider using GPU if available
3. **Model Download Issues**: Ensure stable internet connection (GPT-2 is ~500MB)

### Performance Tips

- Use GPU when available for faster inference
- Lower temperature (0.1-0.5) for more deterministic output
- Higher temperature (0.7-1.0) for more creative output
- Adjust `max-new-tokens` based on your needs and system capacity
- GPT-2 works well with most prompt types - try storytelling, questions, or completions

## License

This project is provided as-is for educational and research purposes. Please check the [GPT-2 model license](https://huggingface.co/gpt2) for model usage terms.

## Contributing

Feel free to submit issues, feature requests, or improvements to this inference script.
