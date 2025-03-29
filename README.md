<div align="center">
  <h1>💫 StarVector: Generating Scalable Vector Graphics Code from Images and Text</h1>
  <img src="assets/starvector-xyz.png" alt="starvector" style="width: 800px; display: block; margin-left: auto; margin-right: auto;"/>

<a href="https://arxiv.org/abs/2312.11556" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-StarVector-red?logo=arxiv" height="25" />
</a>
<a href="https://starvector.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-starvector.github.io-blue.svg" height="25" />
</a>
<a href="https://huggingface.co/starvector/starvector-1b-im2svg" target="_blank">
    <img alt="HF Models: StarVector" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-StarVector--1B-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/starvector/starvector-8b-im2svg" target="_blank">
    <img alt="HF Models: StarVector" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-StarVector--8B-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/datasets/starvector/svg-stack" target="_blank">
    <img alt="HF Dataset: SVG-Stack" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Data-SVG--Stack-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/collections/starvector/starvector-svg-datasets-svg-bench-67811204a76475be4dd66d09" target="_blank">
    <img alt="HF Dataset: SVG-Bench" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-SVG--Bench-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://colab.research.google.com/github/joanrod/star-vector/blob/main/starvector_colab_demo.ipynb" target="_blank">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg" height="25" />
</a>

<div style="font-family: charter;">
    <a href="https://joanrod.github.io" target="_blank">Juan A. Rodriguez</a>,
    <a href="https://abhaypuri.github.io/portfolio/" target="_blank">Abhay Puri</a>,
    <a href="https://shubhamagarwal92.github.io/" target="_blank">Shubham Agarwal</a>,
    <a href="https://scholar.google.ca/citations?user=8vRS7F0AAAAJ&hl=en" target="_blank">Issam H. Laradji</a>,
    <a href="https://scholar.google.es/citations?user=IwBx73wAAAAJ&hl=ca" target="_blank">Pau Rodriguez</a>,
    <a href="https://scholar.google.es/citations?user=1jHvtfsAAAAJ&hl=ca" target="_blank">David Vazquez</a>,
    <a href="https://scholar.google.com/citations?user=1ScWJOoAAAAJ&hl=en" target="_blank">Chris Pal</a>,
    <a href="https://scholar.google.com/citations?user=aVfyPAoAAAAJ&hl=en" target="_blank">Marco Pedersoli</a>
</div>

</div>

## 🔥 News
- March 2025: **StarVector Accepted at CVPR 2025**,
  - StarVector has been accepted at CVPR 2025! Check out the paper [[Link](https://arxiv.org/abs/2312.11556)]
  - Check out our website for more information [[Link](https://starvector.github.io/)]
  - StarVector models are now available on HuggingFace! [[Link](https://huggingface.co/starvector/starvector-1b-im2svg)] [[Link](https://huggingface.co/starvector/starvector-8b-im2svg)]
  - SVGBench and SVG-Stack datasets are now available on HuggingFace Datasets! [[Link](https://huggingface.co/datasets/starvector/svg-bench)] [[Link](https://huggingface.co/datasets/starvector/svg-stack)]
  - Try StarVector in your browser with our Colab notebook! [[Link](https://colab.research.google.com/github/joanrod/star-vector/blob/main/starvector_colab_demo.ipynb)]
  
## 🚀 Introduction
StarVector is a multimodal vision-language model for Scalable Vector Graphics (SVG) generation. It can be used to perform image2SVG and text2SVG generation. We pose image generation as a code generation task, using the power of multimodal VLMs

<div align="center">
  <img src="assets/starvector-teaser.png" alt="starvector" style="width: 900px; display: block; margin-left: auto; margin-right: auto;" />
</div>

> **Abstract**: Scalable Vector Graphics (SVGs) are vital for modern image rendering due to their scalability and versatility. Previous SVG generation methods have focused on curve-based vectorization, lacking semantic understanding, often producing artifacts, and struggling with SVG primitives beyond \textit{path} curves. To address these issues, we introduce StarVector, a multimodal large language model for SVG generation. It performs image vectorization by understanding image semantics and using SVG primitives for compact, precise outputs. Unlike traditional methods, StarVector works directly in the SVG code space, leveraging visual understanding to apply accurate SVG primitives. To train StarVector, we create SVG-Stack, a diverse dataset of 2M samples that enables generalization across vectorization tasks and precise use of primitives like ellipses, polygons, and text. We address challenges in SVG evaluation, showing that pixel-based metrics like MSE fail to capture the unique qualities of vector graphics. We introduce SVG-Bench, a benchmark across 10 datasets, and 3 tasks: Image-to-SVG, Text-to-SVG generation, and diagram generation. Using this setup, StarVector achieves state-of-the-art performance, producing more compact and semantically rich SVGs.

### Multimodal Architecture

StarVector uses a multimodal architecture to process images and text. When performing Image-to-SVG (or image vectorization), the image is projected into visual tokens, and SVG code is generated. When performing Text-to-SVG, the model only recieves the text instruction (no image is provided), and a novel SVG is created. The LLM is based of StarCoder, which we leverage to transfer coding skills to SVG generation.

<div align="center">
  <img src="assets/starvector-arch.png" alt="starvector" style="width: 700px; display: block; margin-left: auto; margin-right: auto;" />
</div>

## 📖 Table of Contents
- [💿 Installation](#installation)
- [🏎️ Quick Start - Image2SVG Generation](#quick-start---image2svg-generation)
- [🎨 Models](#models)
- [📊 Datasets](#datasets---svg-bench)
- [🏋️‍♂️ Training](#training)
- [🏆 Evaluation on SVG-Bench](#validation-on-svg-benchmarks-svg-bench)
- [🧩 Demo](#starvector-demo)
- [📚 Citation](#citation)
- [📝 License](#license)


## Installation

1. Clone this repository and navigate to star-vector folder
```bash
git clone https://github.com/joanrod/star-vector.git
cd star-vector
```

2. Install Package
```Shell
conda create -n starvector python=3.11.3 -y
conda activate starvector
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training
```
pip install -e ".[train]"
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .
```

## Quick Start - Image2SVG Generation

```Python
from PIL import Image
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg

model_name = "starvector/starvector-8b-im2svg"

starvector = StarVectorForCausalLM.from_pretrained(model_name)

starvector.cuda()
starvector.eval()

image_pil = Image.open('assets/examples/sample-0.png')
image = starvector.process_images([image_pil])[0].cuda()
batch = {"image": image}

raw_svg = starvector.generate_im2svg(batch, max_length=1000)[0]
svg, raster_image = process_and_rasterize_svg(raw_svg)
```

### Use it from HuggingFace AutoModel

```Python
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from starvector.data.util import process_and_rasterize_svg
import torch

model_name = "starvector/starvector-8b-im2svg"

starvector = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
processor = starvector.model.processor
tokenizer = starvector.model.svg_transformer.tokenizer

starvector.cuda()
starvector.eval()

image_pil = Image.open('assets/examples/sample-18.png')

image = processor(image_pil, return_tensors="pt")['pixel_values'].cuda()
if not image.shape[0] == 1:
    image = image.squeeze(0)
batch = {"image": image}

raw_svg = starvector.generate_im2svg(batch, max_length=4000)[0]
svg, raster_image = process_and_rasterize_svg(raw_svg)
```


## Models

We provide [Hugging Face 🤗 model checkpoints](https://huggingface.co/collections/starvector/starvector-models-6783b22c7bd4b43d13cb5289) for image2SVG vectorization, for 💫 StarVector-8B and 💫 StarVector-1B. These are the results on SVG-Bench, using the DinoScore metric.

| Method        | SVG-Stack | SVG-Fonts | SVG-Icons | SVG-Emoji | SVG-Diagrams |
|---------------|-----------|-----------|-----------|-----------|--------------|
| AutoTrace    | 0.942     | 0.954     | 0.946     | 0.975     | 0.874        |
| Potrace      | 0.898     | 0.967     | 0.972     | 0.882     | 0.875        |
| VTracer      | 0.954     | 0.964     | 0.940     | 0.981     | 0.882        |
| Im2Vec        | 0.692     | 0.733     | 0.754     | 0.732     | -            |
| LIVE          | 0.934     | 0.956     | 0.959     | 0.969     | 0.870        |
| DiffVG        | 0.810     | 0.821     | 0.952     | 0.814     | 0.822        |
| GPT-4-V       | 0.852     | 0.842     | 0.848     | 0.850     | -            |
| 💫 StarVector-1B (🤗 [Link](https://huggingface.co/starvector/starvector-1b-im2svg)) | 0.926     | 0.978     | 0.975     | 0.929     | 0.943        |
| 💫 StarVector-8B (🤗 [Link](https://huggingface.co/starvector/starvector-8b-im2svg)) | **0.966** | **0.982** | **0.984** | **0.981** | **0.959**    |

*Note*: StarVector models will not work for natural images or illustrations, as they have not been trained on those images. They excel in vectorizing icons, logotypes, technical diagrams, graphs, and charts.

## Datasets - SVG-Bench
SVG-Bench is a benchmark for evaluating SVG generation models. It contains 10 datasets, and 3 tasks: Image-to-SVG, Text-to-SVG, and Diagram-to-SVG.

See our [Huggingface 🤗 Dataset Collection](https://huggingface.co/collections/starvector/starvector-svg-datasets-67811204a76475be4dd66d09)  

| Dataset         |  Train  | Val   | Test | Token Length     | SVG Primitives | Annotation     |
|-----------------|--------|-------|------|------------------|----------------|----------------|
| SVG-Stack (🤗 [Link](https://huggingface.co/datasets/starvector/svg-stack)) | 2.1M   | 108k  | 5.7k | 1,822 ± 1,808    | All            | [Captions](https://huggingface.co/datasets/starvector/text2svg-stack)        |
| SVG-Stack_sim (🤗 [Link](https://huggingface.co/datasets/starvector/svg-stack-simple)) | 601k   | 30.1k | 1.5k | 2k ± 918         | Vector path    | -        |
| SVG-Diagrams (🤗 [Link](https://huggingface.co/datasets/starvector/svg-diagrams)) | -      | -     | 472  | 3,486 ± 1,918    | All            | -        |
| SVG-Fonts (🤗 [Link](https://huggingface.co/datasets/starvector/svg-fonts)) | 1.8M   | 91.5k | 4.8k | 2,121 ± 1,868    | Vector path    | Font letter      |
| SVG-Fonts_sim (🤗 [Link](https://huggingface.co/datasets/starvector/svg-fonts-simple)) | 1.4M   | 71.7k | 3.7k | 1,722 ± 723      | Vector path    | Font letter      |
| SVG-Emoji (🤗 [Link](https://huggingface.co/datasets/starvector/svg-emoji)) | 8.7k   | 667   | 668  | 2,551 ± 1,805    | All            | -          |
| SVG-Emoji_sim (🤗 [Link](https://huggingface.co/datasets/starvector/svg-emoji-simple)) | 580    | 57    | 96   | 2,448 ± 1,026    | Vector Path    | -          |
| SVG-Icons (🤗 [Link](https://huggingface.co/datasets/starvector/svg-icons)) | 80.4k  | 6.2k  | 2.4k | 2,449 ± 1,543    | Vector path    | -              |
| SVG-Icons_sim (🤗 [Link](https://huggingface.co/datasets/starvector/svg-icons-simple)) | 80,435 | 2,836 | 1,277| 2,005 ± 824      | Vector path    | -              |
| SVG-FIGR (🤗 [Link](https://huggingface.co/datasets/starvector/FIGR-SVG)) | 270k   | 27k   | 3k   | 5,342 ± 2,345    | Vector path    | Class, Caption | 


>We offer a summary of statistics about the datasets used in our training and evaluation experiments. This datasets are included in SVG-Bench. The subscript _sim_ stands for the simplified version of the dataset, as required by some baselines.

## Training

### Confirm dependencies are installed

```bash
pip install -e ".[train]"
```

### Set environment variables
We recommend setting the following environment variables:

```bash
  export HF_HOME=<path to the folder where you want to store the models>
  export HF_TOKEN=<your huggingface token>
  export WANDB_API_KEY=<your wandb token>
  export OUTPUT_DIR=<path/to/output>
```