# GPT4Image: Can Large Pre-trained Models Help Vision Models on Perception Tasks?

PyTorch implementation for arxiv paper : https://arxiv.org/abs/2306.00693

Image description set and usage : https://dingning97.github.io/imagenet-descriptions/

<img src="https://dingning97.github.io/imagenet-descriptions/assets/img/framework.png" width="900">

## Data Preparation

(1)  Prepare for ImageNet dataset: \
Download and extract ImageNet train and val images from http://image-net.org/.
The training and validation data is expected to be in the `train` folder and `val` folder respectively:
```
/path/to/imagenet
  ├── train
  │     ├── class1
  │     │      ├── img1.jpeg
  │     ├── class2
  │            ├──img2.jpeg
  │
  └─── val
        ├── class1
        │      ├──img3.jpeg
        ├── class2
               ├──img4.jpeg
```

(2)  Prepare for text embeddings used for training:\
Download the image description set at this [link](https://dingning97.github.io/imagenet-descriptions/) and place the file "minigpt4_caption_imagenet_train_0_1281166.pth" in this folder.\
Install CLIP according to [official instruction](https://github.com/openai/CLIP).
```bash
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```
Run the script "description2embedding.py" to convert the image descriptions to embeddings.
```bash
$ python description2embedding.py --caption=./minigpt4_caption_imagenet_train_0_1281166.pth
```
You will have "imagenet_clip_text_emb_0_1281166.pth" generated in this folder.

## Train ResNet-50
You need 4 Tesla-V100 GPUs to run
```bash
$ cd resnet
$ pip install -r requirements.txt
$ python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --data=/path/to/imagenet --batch-size=256 --text_emb=imagenet_clip_text_emb_0_1281166.pth --save_dir=./outputs [--use_amp]
```
You will get the result max top-1 acc ~72.8%

## Train DeiT-base
You need 8 Tesla-V100 GPUs to run
```bash
$ cd deit
$ pip install -r requirements.txt
$ python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --data-path=/path/to/imagenet --batch-size=128 --text_emb=imagenet_clip_text_emb_0_1281166.pth --output_dir=./outputs
```
You will get the result max top-1 acc ~82.3%