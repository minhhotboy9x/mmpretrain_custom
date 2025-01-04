import torch
from mmengine.config import Config
from mmpretrain.registry import MODELS
from mmengine.config import read_base


# x = torch.zeros(3, 5)
# index = torch.tensor([[0, 1, 2, 0, 1, 0],
#                       [1, 2, 0, 2, 1, 1],
#                       [2, 0, 1, 2, 0, 2]])
# src = torch.tensor([[1, 2, 3, 4, 5, 6],
#                     [5, 4, 3, 2, 1, 6],
#                     [1, 2, 3, 4, 5, 6]], dtype=torch.float32)

# # Gán giá trị từ src vào x tại các chỉ số trong index dọc theo dim=1
# x.scatter_(dim=1, index=index, src=src)

# print(x)

# y = torch.tensor([[1, 2, 3, 4, 5, 6], [5, 4, 3, 2, 1, 6], [1, 2, 3, 4, 5, 6]], dtype=torch.float32)

# print(y.repeat(2, 1))

# default_scope = 'mmpretrain' # need to set

# cfg = Config.fromfile('./customs/aggregation_configs/config_aggregation_sanbox.py')
cfg = Config.fromfile('customs/aggregation_configs/baseline1_resnet50_2stages_malaria__pa5_2+5_class.py')

model = MODELS.build(cfg.model)

# Kiểm tra mô hình
print(model)

print('-------------------------')
# Tạo dữ liệu giả lập

dummy_input = torch.randn(2, 3, 224, 224)

# Forward dữ liệu qua mô hình
output = model(dummy_input)

# Kiểm tra đầu ra
print(output)