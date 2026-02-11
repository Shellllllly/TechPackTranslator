# TechPackTranslator
This is a demo translator specifically for translating texts from English to Chinese in tech pack images, while remaining the same structure of the image.


## Overview
The system design(pipeline) can be found in `algorithm_pipeline.mmd`. (Mermaid plugin/extension needed)


## Main Functions
- OCR: extract original texts using EasyOCR.
- Translation: translate english to Chinese(traditional) using Microsoft Phi4.
- Results saving: 1. Maintain the original structure - overwrite the original text - using Pillow; 2. Save to text file;
- containerization: support to run in docker container. 


## Dependence
- Please see the `requirements.txt`. 
- Nvidia GPU

## Usage
1. Clone the reposipory.
2. Build Docker image: 
```docker build -t tech-pack-translator .```
3. Create a container and execute the translator with required argument: 
```docker run --gpus=all -v $(pwd)/test_data:/app/test_data tech-pack-translator python main.py --input test_data/test.jpeg```

