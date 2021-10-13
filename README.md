
# yolov5模型剪枝
首先使用基于yolov5m731.pt进行稀疏训练：

```
python train_sparsity.py --st --src 0.0001 --srb 0.0001 --weights yolov5m731.pt --adam --epochs 100
```

src、srb的选择需要根据数据集调整，可以通过观察tensorboard的map，gamma变化直方图等选择。
在run/train/exp*/目录下:
```
tensorboard --logdir .
```
然后点击出现的链接观察训练中的各项指标.

训练完成后进行剪枝：

```
python prune_convbn.py --weights runs/train/exp1/weights/last.pt --conv_percent 0.15 --bn_percent 0.75
```

裁剪比例conv_percent、bn_percent根据效果调整，可以从小到大试。裁剪完成会保存对应的模型pruned_model.pt。

微调：

```
python finetune_prune_conv.py --weights pruned_model.pt --adam --epochs 100
```



| model                 | pruned | map   | mode size |
| --------------------- | ------- | ----- | --------- |
| yolov5m731.pt         | -       | 80.0 | 40.6 M    |
| sparity.pt | 0.15\0.75   | 74.6 | 40.6 M    |
| pruned.pt    | -   | 74.6 | 16.0 M     |
| fine-tune.pt             | -       | 70.0 | 8.2 M     |


## 调参
1. 浅层尽量少剪,从训练完成后gamma每一层的分布也可以看出来.
2. 系数λ的选择需要平衡map和剪枝力度.首先通过train.py训练一个正常情况下的baseline.然后在稀疏训练过程中观察MAP和gamma直方图变化,MAP掉点严重和gamma稀疏过快等情况下,可以适当降低λ.反之如果你想压缩一个尽量小的模型,可以适当调整λ.
3. 稀疏训练=>剪枝=>微调 可以反复迭代这个过程多次剪枝.

## 常见问题
1. 稀疏训练是非常种重要的,也是调参的重点,多观察bn直方图变化,过快或者过慢都不适合,所以需要平衡你的sr, lr等.一般情况下,稀疏训练的结果和正常训练map是比较接近的.
2. 剪枝时候多试试不同的ratio,一个基本的准则是每层bn层至少保留一个channel,所以有时候稀疏训练不到位,而ratio设置的很大,会看到remaining channel里面会有0出现,这时候要么设置更小的ratio,要么重新稀疏训练,获得更稀疏的参数.
3. 如果想要移植到移动端，可以使用ncnn加速，另外剪枝时控制剩余channel为2^n能有效提升推理速度；GPU可以使用TensorRT加速。
