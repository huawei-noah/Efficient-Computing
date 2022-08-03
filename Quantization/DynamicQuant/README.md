# Dynamic Quantization

Instance-Aware Dynamic Neural Network Quantization, CVPR 2022.

By Zhenhua Liu, Yunhe Wang, Kai Han, Siwei Ma and Wen Gao.

<img width="518" alt="搜狗截图22年08月03日1525_1" src="https://user-images.githubusercontent.com/19202799/182549238-2cc1db63-e504-483f-8a2e-ff51d94974cb.png">


## Requirements
Pytorch 1.7.0

## Usage

To train the model:
```
python train.py /path/to/imagenet --arch resnet18 --tar_bit 4
```

To evaluate the model:
```
python train.py /path/to/imagenet --resume /path/to/resume --arch resnet18 --tar_bit 4 --evaluate
```

## Results

| Model | Bit | DoReFa | DoReFa+DQ |
|-- | --| --|--|
| ResNet-18| 4 | 68.1 | 69.23 |

Download checkpoints: [BaiduDisk](https://pan.baidu.com/s/1VrXoFBL78x0_a_67y6j_Xg), passward: [lly0]().

## Citations

	@inproceedings{DynQuant,
		title={Instance-Aware Dynamic Neural Network Quantization},
		author={Liu, Zhenhua and Wang, Yunhe and Han, Kai and Ma, Siwei and Gao, Wen},
		booktitle={CVPR},
		year={2022}
	}
