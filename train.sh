#!/usr/bin/env sh
#nohup python train.py --img 640 --batch 4 --epochs 100 --data coco128.yaml --weights yolov5m.pt >> log0729.txt
#python train.py --img 640 --batch 4 --epochs 100 --data coco128.yaml --cfg models/yolov5s.yaml

echo "train begin..."
nohup python train_sparsity4.py > 1012-100-1epoch.log 2>&1 &
#nohup python train_prun.py > ./0831-100-prun.log 2>&1 &
