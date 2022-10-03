# AdaBin: Improving Binary Neural Networks with Adaptive Binary Sets
By Zhijun Tu, Xinghao Chen, Pengju Ren and Yunhe Wang

This is the PyTorch implementation of ECCV 2022 paper "AdaBin: Improving Binary Neural Networks with Adaptive Binary Sets” . [[arXiv]](https://arxiv.org/abs/2208.08084)

The mindspore code has also been released: [AdaBin](https://gitee.com/mindspore/models/tree/master/research/cv/AdaBin)

## Requirements
````
torch==1.8.0
torchvision==0.9.0
prefetch_generator
progress
````

## Results
-  Classification results on CIFAR-10

| Model | Bit-width (W/A) | Accuracy |
| --- | --- | --- |
| ResNet-20 | 1/1 | 88.1% |
| ResNet-18 | 1/1 | 62.1% |
| VGG-small | 1/1 | 92.3% |

-  Classification results on ImageNet-1k (* means using the two-step training setting as ReActNet)

| Model      | Bit-width (W/A) | Top-1. Acc | Top-5. Acc |
| ---------- | --------------- | ---------- | ---------- |
| AlexNet    | 1/1             | 53.9%      | 77.6%      |
| ResNet-18  | 1/1             | 63.1%      | 84.3%      |
| ResNet-18* | 1/1             | 66.4%      | 86.5%      |
| ResNet-34  | 1/1             | 66.4%      | 86.6%      |

## Citation

    @inproceedings{tu2022adabin,
        title={AdaBin: Improving Binary Neural Networks with Adaptive Binary Sets},
        author={Zhijun Tu, Xinghao Chen, Pengju Ren and Yunhe Wang},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2022}
    }
