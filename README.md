# StyleEval
This is the official implementation of paper "[Evaluate and improve the quality of neural style transfer](https://www.researchgate.net/publication/350184156_Evaluate_and_improve_the_quality_of_neural_style_transfer)" (CVIU 2021))

### Environment Implemented:
- Python 3.6
- TensorFlow 1.4.0
- CUDA 8.0

### Getting Start

Step 1: clone this repo


`git clone https://github.com/EndyWon/StyleEval`  
`cd StyleEval`


Step 2: download pre-trained vgg19 model


`bash download_vgg19.sh`

Step 3:  test on single image pair

`python quality_criteria.py --content ContentImagePath --style StyleImagePath --stylized StylizedImagePath`

### Discussions

We tested images of size 512x512 on an Intel Core i7-5820K CPU @ 3.30GHz × 12, the average time is about 126 seconds. We found that the most time was spent on calculating LP scores (about 80 seconds).

Currently the evaluation time is too slow, please consider to use GPU convolution to accelerate the patch matching process in LP calculation.
