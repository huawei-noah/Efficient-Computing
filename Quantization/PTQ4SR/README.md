# Toward Accurate Post-Training Quantization for Image Super Resolution
By Zhijun Tu, Jie Hu, Hanting Chen, Yunhe Wang

This is the PyTorch implementation of CVPR 2023 paper [Toward Accurate Post-Training Quantization for Image Super Resolution](https://openaccess.thecvf.com/content/CVPR2023/papers/Tu_Toward_Accurate_Post-Training_Quantization_for_Image_Super_Resolution_CVPR_2023_paper.pdf) based on BasicSR. 

The MindSpore code has also been released: [PTQ4SR](https://gitee.com/torch/models/tree/master/research/cv/PTQ4SR).

## Requirements
````
addict
future
lmdb
numpy>=1.17
opencv-python
Pillow
pyyaml
requests
scikit-image
scipy
tb-nightly
tqdm
yapf
wheel==0.26
````
## Scripts
````
# EDSR_Lx4
python basicsr/ptq4sr.py -opt options/ptq/EDSR/test_EDSR_Lx4_ptq_w8a8.yml
python basicsr/ptq4sr.py -opt options/ptq/EDSR/test_EDSR_Lx4_ptq_w6a6.yml
python basicsr/ptq4sr.py -opt options/ptq/EDSR/test_EDSR_Lx4_ptq_w4a4.yml

# EDSR_Lx2
python basicsr/ptq4sr.py -opt options/ptq/EDSR/test_EDSR_Lx2_ptq_w8a8.yml
python basicsr/ptq4sr.py -opt options/ptq/EDSR/test_EDSR_Lx2_ptq_w6a6.yml
python basicsr/ptq4sr.py -opt options/ptq/EDSR/test_EDSR_Lx2_ptq_w4a4.yml

# SRResNet_x4
python basicsr/ptq4sr.py -opt options/ptq/SRResNet/test_MSRResNet_x4_ptq_w8a8.yml
python basicsr/ptq4sr.py -opt options/ptq/SRResNet/test_MSRResNet_x4_ptq_w6a6.yml
python basicsr/ptq4sr.py -opt options/ptq/SRResNet/test_MSRResNet_x4_ptq_w4a4.yml

# SRResNet_x4
python basicsr/ptq4sr.py -opt options/ptq/SRResNet/test_MSRResNet_x2_ptq_w8a8.yml
python basicsr/ptq4sr.py -opt options/ptq/SRResNet/test_MSRResNet_x2_ptq_w6a6.yml
python basicsr/ptq4sr.py -opt options/ptq/SRResNet/test_MSRResNet_x2_ptq_w4a4.yml
````


## Citation

    @inproceedings{tu2023toward,
        title={Toward accurate post-training quantization for image super resolution},
        author={Tu, Zhijun and Hu, Jie and Chen, Hanting and Wang, Yunhe},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={5856--5865},
        year={2023}
}
