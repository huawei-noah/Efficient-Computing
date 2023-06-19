## Masked Image Modeling with Local Multi-Scale Reconstruction
PyTorch implementation of
<br>
[**Masked Image Modeling with Local Multi-Scale Reconstruction**](https://arxiv.org/abs/2303.05251)
<br>
CVPR 2023

<p align="center">
  <img src="model.png" width="1000">
</p>

Masked Image Modeling (MIM) achieves outstanding success in self-supervised representation learning. Unfortunately, MIM models typically have huge computational burden and slow learning process, which is an inevitable obstacle for their industrial applications. Although the lower layers play the key role in MIM, existing MIM models conduct reconstruction task only at the top layer of encoder. The lower layers are not explicitly guided and the interaction among their patches is only used for calculating new activations. Considering the reconstruction task requires non-trivial inter-patch interactions to reason target signals, we apply it to multiple local layers including lower and upper layers. Further, since the multiple layers expect to learn the information of different scales, we design local multi-scale reconstruction, where the lower and upper layers reconstruct fine-scale and coarse-scale supervision signals respectively. This design not only accelerates the representation learning process by explicitly guiding multiple layers, but also facilitates multi-scale semantical understanding to the input. Extensive experiments show that with significantly less pre-training burden, our model achieves comparable or better performance on classification, detection and segmentation tasks than existing MIM models.

### Pre-Trained Models

| Backbone | #Params | Target | GPU Hours/Ep. | PT Epoch | PT Resolution |             PT log/ckpt              | Top-1 (%) |
|:---------|:-------:|:------:|:-------------:|:--------:|:-------------:|:------------------------------------:|:---------:|
| ViT-B    |   86M   |  HOG   |      0.7      |   1600   |    224x224    |      [log](https://github.com/Haoqing-Wang/LocalMIM/releases/download/pretrain/vit_base_localmim_hog_1600ep_pretrain.txt)/[ckpt](https://github.com/Haoqing-Wang/LocalMIM/releases/download/pretrain/vit_base_localmim_hog_1600ep_pretrain.pth)      |   84.0    |
| ViT-L    |  307M   |  HOG   |      1.0      |   800    |    224x224    |      [log](https://github.com/Haoqing-Wang/LocalMIM/releases/download/pretrain/vit_large_localmim_hog_800ep_pretrain.txt)/[ckpt](https://github.com/Haoqing-Wang/LocalMIM/releases/download/pretrain/vit_large_localmim_hog_800ep_pretrain.pth)      |   85.8    |
| Swin-B   |   88M   | Pixel  |      1.0      |   400    |    224x224    |      [log](https://github.com/Haoqing-Wang/LocalMIM/releases/download/pretrain/swin_base_localmim_pixel_400ep_pretrain.txt)/[ckpt](https://github.com/Haoqing-Wang/LocalMIM/releases/download/pretrain/swin_base_localmim_pixel_400ep_pretrain.pth)      |   84.0    |
| Swin-B   |   88M   |  HOG   |      1.1      |   400    |    224x224    |      [log](https://github.com/Haoqing-Wang/LocalMIM/releases/download/pretrain/swin_base_localmim_hog_400ep_pretrain.txt)/[ckpt](https://github.com/Haoqing-Wang/LocalMIM/releases/download/pretrain/swin_base_localmim_hog_400ep_pretrain.pth)      |   84.1    |
| Swin-L   |  197M   |  HOG   |      1.6      |   800    |    224x224    |      [log](https://github.com/Haoqing-Wang/LocalMIM/releases/download/pretrain/swin_large_localmim_hog_800ep_pretrain.txt)/[ckpt](https://github.com/Haoqing-Wang/LocalMIM/releases/download/pretrain/swin_large_localmim_hog_800ep_pretrain.pth)      |   85.6    |

The pre-training and fine-tuning instruction can be found in [ViT](ViT/README.md), [Swin](Swin/README.md) and [semantic_segmentation](semantic_segmentation/README.md).

### Citation
If you find this project useful in your research, please consider cite:
```
@inproceedings{wang2023masked,
  title={Masked Image Modeling with Local Multi-Scale Reconstruction},
  author={Wang, Haoqing and Tang, Yehui and Han, Kai and Guo, Jianyuan and Deng, Zhi-Hong and Wang, Yunhe},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={xxx--xxx},
  year={2023}
}
```

### Acknowledgement

This code is built upon the implementation from [MAE](https://github.com/facebookresearch/mae), [GreenMIM](https://github.com/LayneH/GreenMIM), [MMSeg](https://github.com/open-mmlab/mmsegmentation) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit).
