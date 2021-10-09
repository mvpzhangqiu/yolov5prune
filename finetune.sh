#!/usr/bin/env sh
#nohup python train.py --img 640 --batch 4 --epochs 100 --data coco128.yaml --weights yolov5m.pt >> log0729.txt
#python train.py --img 640 --batch 4 --epochs 100 --data coco128.yaml --cfg models/yolov5s.yaml

echo "train begin..."
nohup python finetune_pruned.py --epochs 30 > 0930-50-finetune.log 2>&1 &
