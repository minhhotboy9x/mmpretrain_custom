import torch
from mmengine.config import Config
from mmpretrain.registry import MODELS
from mmengine.config import read_base


# Tạo tensor 1 chiều
# tensor = torch.tensor([3, 7, 1, 9, 7, 5])

# # Giá trị cần tìm index
# value = 0

# # Lấy index của phần tử
# indices = torch.where(tensor == value)[0].item()

# print(indices)  # Output: tensor([1, 4])


default_scope = 'mmpretrain' # need to set

# cfg = Config.fromfile('./customs/aggregation_configs/config_aggregation_sanbox.py')
cfg = Config.fromfile('customs/aggregation_configs/baseline1_resnet50_2stages_malaria_pa5_2+5_class.py')

model = MODELS.build(cfg.model)

# Kiểm tra mô hình
print(model)

print('-------------------------')
# Tạo dữ liệu giả lập

dummy_input = torch.randn(1, 3, 224, 224)

# Forward dữ liệu qua mô hình
output = model(dummy_input, mode = 'predict')

# Kiểm tra đầu ra
print(output)