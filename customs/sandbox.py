from mmengine.config import Config
from mmengine import MODELS

# cfg = Config.fromfile('./customs/config_aggregation.py')
cfg = Config.fromfile('./customs/config_aggregation.py')

model = MODELS.build(cfg.model)

# Kiểm tra mô hình
print(model)