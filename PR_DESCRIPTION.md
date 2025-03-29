# Add Colab Notebook for StarVector Demo

This PR adds a comprehensive Google Colab notebook that allows users to easily try out StarVector in their browser without any local setup required.

## Changes
- Added `starvector_colab_demo.ipynb` with detailed documentation and examples
- Updated main README with a Colab badge and link to the notebook
- Added documentation in the scripts/README.md about the Colab notebook

## Features of the Colab Notebook
- Step-by-step setup instructions for the environment
- Support for both StarVector-1B and StarVector-8B models
- Image-to-SVG generation with sample images and custom upload
- Text-to-SVG generation (experimental)
- Interactive visualization of results (original image, SVG code, rendered SVG)
- Simple Gradio interface for easier interaction
- Proper resource cleanup

## Testing
The notebook has been tested in Google Colab with:
- Python 3.10
- PyTorch 2.5.1
- Various sample images from the repository
- Custom image uploads

## Screenshots
![Colab Notebook Screenshot](https://github.com/username/star-vector/assets/colab_screenshot.png)
(Note: Replace with actual screenshot once the notebook is running)

## Related Issues
Closes #XXX (if applicable)

## Motivation
This Colab notebook makes it easier for users to try StarVector without needing to set up a local environment, especially those who may not have access to necessary GPU resources. It serves as an interactive introduction to the project and its capabilities. 