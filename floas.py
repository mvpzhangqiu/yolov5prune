from models.experimental import attempt_load
from utils import torch_utils
from models import experimental

weights_origin = '/home/zq/work/test/yolov5m-7.31.pt'
# weights_pruned = 'runs/train/exp10/weights/best.pt'
weights_pruned = 'runs/weights/spar43/finetune_model.pt'
print("origin:")
model = attempt_load(weights_origin)
torch_utils.model_info(model)
print("=" * 150)
print("pruned:")
model = attempt_load(weights_pruned)
torch_utils.model_info(model)