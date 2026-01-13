# VAEVQ: Enhancing Discrete Visual Tokenization through Variational Modeling

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2511.06863-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2511.06863)


</h5>

## Implementations

### Installation

- **Dependencies**: `pip install -r requirements.txt`
- **Datasets**
```
imagenet
└── train/
    ├── n01440764
        ├── n01440764_10026.JPEG
        ├── n01440764_10027.JPEG
        ├── ...
    ├── n01443537
    ├── ...
└── val/
    ├── ...
```

### Training Scripts
* Image Tokenizer Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py fit --config ./configs/imagenet_vaevq.yaml 
```

### Evaluation Scripts
* Image Tokenizer Evaluation
```
python eval.py --config "./configs/imagenet_vaevq.yaml" --ckpt_path "./vq_log/xxx/epoch=49-step=250250.ckpt" 
```

## Acknowledgement
The codebase of VAEVQ is adapted from [SimVQ](https://github.com/youngsheen/SimVQ). Thanks for their wonderful work.


## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{yang2025vaevq,
  title={VAEVQ: Enhancing Discrete Visual Tokenization through Variational Modeling},
  author={Yang, Sicheng and Hu, Xing and Wu, Qiang and Yang, Dawei},
  journal={arXiv preprint arXiv:2511.06863},
  year={2025}
}
```