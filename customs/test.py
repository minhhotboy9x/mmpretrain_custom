import os

from mmengine.runner import Runner
from mmengine.config import Config
from mmengine.evaluator import DumpResults
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Actual labels')
    plt.ylabel('Predicted labels')
    plt.tight_layout()

    # Nếu có đường dẫn lưu (save_path) thì lưu confusion matrix vào file
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

def acc_each_label(confusion_matrix):
    acc = []
    for i in range(len(confusion_matrix)):
        acc.append(confusion_matrix[i][i] / sum(confusion_matrix[i]))
    return acc

if __name__ == '__main__':

    cfg = Config.fromfile('customs/aggregation_configs/baseline1_resnet50_2stages_malaria_pa5_2+5_class_oversample.py')
    # cfg = Config.fromfile('customs/aggregation_configs/config_aggregation_sanbox.py')

    cfg.work_dir = './work_dirs/my_sandbox'
    cfg.load_from = './work_dirs/experiment_result_pa5_2stages_res50_oversamplex8/best_accuracy_top1_epoch_65.pth'
    cfg.test_cfg.fp16 = True
    cfg.default_hooks.visualization.enable = True
    cfg.default_hooks.visualization.interval = 1
    cfg.default_hooks.visualization.show = False
    cfg.default_hooks.visualization.wait_time = 2
    cfg.default_hooks.visualization.out_dir = os.path.join(cfg.work_dir, 'visualization')
    cfg.default_hooks.visualization.interval = 1

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    print('--------------------------------------------------')
    # runner.test_evaluator.metrics.append(
    #         DumpResults(out_file_path=cfg.work_dir + 'preds.json'))

    confusion_matrix = metrics['confusion_matrix/result'] # row: actual, col: predicted
    # mmengine.dump(metrics['confusion_matrix/result'].tolist(), cfg.work_dir + '/metrics.json')

    names = ('Ring', 'Trophozoite', 'Schizont', 'Gametocyte', 'Unparasitized', 'Diff')
    
    # Đặt đường dẫn nơi bạn muốn lưu confusion matrix
    save_path = cfg.work_dir + '/confusion_matrix.png'
    plot_confusion_matrix(confusion_matrix.T, names, save_path=save_path)
    acc = acc_each_label(confusion_matrix)
    print("Accuracy of each class:")
    print(names)
    print(acc)
