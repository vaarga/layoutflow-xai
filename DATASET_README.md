---
license: apache-2.0
---
This repository holds the dataset and pretrained weights for LayoutFlow ([Paper](https://arxiv.org/abs/2403.18187)|[Code](https://github.com/JulianGuerreiro/LayoutFlow)).

You can download the data as follows:
```
git clone https://huggingface.co/JulianGuerreiro/LayoutFlow
```

## Checkpoints
We provide checkpoints for our proposed approach called LayoutFlow for both the RICO and the PubLayNet dataset. 
We further provide the weights of the same model architecture trained with Diffusion, which we call LayoutDMx.

## Datasets
We provide the RICO and PubLayNet datasets with the same split as in the LayoutDiffusion (Zhang et al., ICCV 2023) paper as `.pt` files. 
For more information on how to extract the data, you can take a look at the dataloader in our GitHub repository.
 
 **RICO**
 - `ldm_rico_(train/val/test).pt`: regular data
 - `ldm_lex_rico_(train/val/test).pt`: data with layout elements sorted in lexographical order (is used by the model if `dataset.lex_order=True`)

**PubLayNet**
 - `publaynet_(train/val/test).pt`: regular data
 - `ldm_lex_publaynet_(train/val/test).pt`: data with layout elements sorted in lexographical order (is used by the model if `dataset.lex_order=True`)
 - `publaynet_(train/val/test)_inoue.pt`: regular data, but split following LayoutDM (Inoue et al., CVPR 2023)