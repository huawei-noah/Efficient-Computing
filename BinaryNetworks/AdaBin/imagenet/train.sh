python main.py --data /cache/imagenet/ \
               --arch resnet18_1w1a \
               --lr 0.1 \
               -c ./checkpoints/resnet18_1w1a \
               --gpu 0,1,2,3,4,5,6,7