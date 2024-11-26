from mmengine.runner import Runner
from mmengine.config import Config


if __name__ == '__main__':
    
    cfg = Config.fromfile('customs/aggregation_configs/baseline1_resnet50_malaria_pa3_7_class.py')
    # cfg = Config.fromfile('customs/aggregation_configs/config_aggregation_sanbox.py')

    cfg.work_dir = './work_dirs/my_sandbox'

    runner = Runner.from_cfg(cfg)
    runner.train()