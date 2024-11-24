import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class GenImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, minor_classes=[], remove_major=False):
        """
        Args:
            annotations_file (str): Đường dẫn tới file txt chứa nhãn.
            img_dir (str): Thư mục chứa các ảnh.
            transform (callable, optional): Các phép biến đổi cho ảnh (nếu cần).
            minor_classes (list): Danh sách các class thuộc nhóm minor.
        """
        # Đọc file nhãn
        with open(annotations_file, 'r') as f:
            self.img_labels = [line.strip().split() for line in f]
        
        if remove_major:
            self.img_labels = [item for item in self.img_labels if int(item[1]) in minor_classes]
            
        self.img_dir = img_dir
        self.transform = transform
        self.minor_classes = minor_classes
        self.indices_by_minor_class = self._group_indices_by_minor_class()

    def _group_indices_by_minor_class(self):
        """Nhóm chỉ số (indices) theo từng class."""
        indices_by_minor_class = {}

        for idx, (_, label) in enumerate(self.img_labels):
            label = int(label)
            if label not in self.minor_classes:
                continue
            if label not in indices_by_minor_class:
                indices_by_minor_class[label] = []
            indices_by_minor_class[label].append(idx)

        return indices_by_minor_class

    def sample_minor_img(self, batch):
        cls_idx = np.random.choice(self.minor_classes)
        batch_idx = np.random.choice(self.indices_by_minor_class[cls_idx], batch)
        sample_img = []
        sample_label = []
        for idx in batch_idx:
            image, label = self.__getitem__(idx)
            sample_img.append(image)
            sample_label.append(label)
        sample_img = torch.stack(sample_img)  # Ghép các ảnh thành tensor
        sample_label = torch.tensor(sample_label)  # Nhãn là tensor 1D

        return sample_img, sample_label
    
    def __len__(self):
        """Trả về số lượng mẫu trong Dataset."""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Lấy một mẫu dữ liệu tại vị trí idx.
        Args:
            idx (int): Chỉ số của mẫu cần lấy.
        Returns:
            Tuple (image, label): Ảnh đã biến đổi và nhãn.
        """
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = Image.open(img_path).convert("RGB")  # Đảm bảo ảnh ở định dạng RGB
        label = int(self.img_labels[idx][1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def build_dataloader(annotations_file, img_dir, transform, batch_size, minor_classes, remove_major=False, num_workers=4, shuffle=True):
    """Hàm tạo DataLoader từ dataset."""
    dataset = GenImageDataset(annotations_file, img_dir, transform=transform, minor_classes=minor_classes, remove_major=remove_major)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Tạo dataset chính
    annotations_file = "dataset/final_malaria_full_class_classification/test_annotation.txt"
    img_dir = "dataset/final_malaria_full_class_classification"
    minor_classes = [0, 1, 2, 3]  # Ví dụ: các class thuộc nhóm minor

    dataset = GenImageDataset(annotations_file, img_dir, 
                             transform=transform, 
                              minor_classes=minor_classes)

    # Tạo DataLoader chính
    main_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    main_iter = iter(main_dataloader)

    for _ in range(5):  # Lấy 5 batch làm ví dụ
        try:
            main_batch = next(main_iter)
        except StopIteration:
            break
        
        main_images, main_labels = main_batch
        minor_images, minor_labels = main_dataloader.dataset.sample_minor_img(main_dataloader.batch_size)

        print("Main batch labels:", main_labels)
        print("Minor batch labels:", minor_labels)
