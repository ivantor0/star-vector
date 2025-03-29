# StarVector Scripts

This directory contains various utility scripts for working with StarVector models.

## Colab Notebook

We provide a Colab notebook that allows you to easily try StarVector in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joanrod/star-vector/blob/main/starvector_colab_demo.ipynb)

The notebook includes:
- Setting up the environment
- Installing StarVector and its dependencies
- Loading pre-trained models
- Image-to-SVG conversion
- Text-to-SVG generation (experimental)
- A simple Gradio interface for interactive use

## QuickStart Scripts

We provide several quickstart scripts to demonstrate how to use StarVector:

- `quickstart.py` - Basic example of image-to-SVG generation using StarVector's native API
- `quickstart-hf.py` - Example using the HuggingFace Transformers interface

## Usage Examples

To run the quickstart script:

```bash
python scripts/quickstart.py
```

To use the HuggingFace interface:

```bash
python scripts/quickstart-hf.py
``` 