from mmengine.runner import Runner
from mmpretrain.utils import register_all_modules
from mmengine.config import Config

register_all_modules()

if __name__ == '__main__':
    
    cfg = Config.fromfile('customs/aggregation_configs/baseline1_resnet50_2stages_malaria_pa5_2+5_class.py')
    # cfg = Config.fromfile('customs/aggregation_configs/config_aggregation_sanbox.py')

    cfg.work_dir = './work_dirs/my_sandbox'

    runner = Runner.from_cfg(cfg)
    runner.train()