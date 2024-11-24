import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from module import GenSMOTE
from sklearn.neighbors import NearestNeighbors
from dataloader import build_dataloader

# ----------------- Hyperparameters ----------------- #
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# train_annotations_file = "dataset/final_malaria_full_class_classification/train_annotation.txt"
# train_img_dir = "dataset/final_malaria_full_class_classification"

# val_annotations_file = "dataset/final_malaria_full_class_classification/val_annotation.txt"
# val_img_dir = "dataset/final_malaria_full_class_classification"

batch_size = 4
epochs = 100
lr0 = 1e-2
lrf = 1e-2
momentum = 0.937 # (float) SGD momentum/Adam beta1
weight_decay = 0.0005 # (float) optimizer weight decay 5e-4
num_workers = 4
args = {
        'dim_h': 64,
        'n_channel': 3,
        'n_z': 256,
        'num_class': 7,
    } 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Thay đổi kích thước ảnh
    transforms.RandomHorizontalFlip(p=0.5), # Lật ngang với xác suất 50%
    transforms.RandomVerticalFlip(p=0.5),   # Lật dọc với xác suất 50%
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Thay đổi màu sắc
    transforms.ToTensor()                   # Chuyển ảnh thành Tensor
])

  
# ----------------- DataLoader ----------------- #
# train_loader = build_dataloader(train_annotations_file,
#                                 train_img_dir,
#                                 transform, 
#                                 batch_size=batch_size, 
#                                 minor_classes=[0, 1, 2, 3],
#                                 remove_major=True,
#                                 num_workers=num_workers,
#                                 shuffle=True)

# val_loader = build_dataloader(val_annotations_file,
#                                 val_img_dir,
#                                 transform=transform,
#                                 batch_size=batch_size, 
#                                 minor_classes=[0, 1, 2, 3],
#                                 remove_major=True,
#                                 num_workers=num_workers,
#                                 shuffle=False)

# ----------------- Criterion ----------------- #
criterion = nn.MSELoss()

# ----------------- Model ----------------- #
model = GenSMOTE(args)

# ----------------- Optimizer ----------------- #

optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=momentum, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr0*lrf)

# ----------------- Training ----------------- #
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, log_dir="work_dirs/gensmote_image_logs"):
    
    # Khởi tạo TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    
    model.to(device)

    # Khởi tạo scaler cho AMP
    scaler = amp.GradScaler()
    best_loss = float('inf') 

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_R_loss = 0.0
        
        # Vòng lặp huấn luyện
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for step, (images, labels) in enumerate(train_loop):
            images, labels = images.to(device), labels.to(device)
            
            # Sử dụng AMP để huấn luyện
            with amp.autocast(device_type='cuda'): 
                # Reconstruction loss
                outputs = model(images, labels)
                R_loss = criterion(outputs, images)
                loss = R_loss
                
            # Backward và cập nhật optimizer
            optimizer.zero_grad()
            
            # Sử dụng GradScaler để tính toán và cập nhật gradient
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Cộng dồn loss để tính trung bình
            running_loss += loss.item()
            running_R_loss += R_loss.item()
            
            # Log loss vào TensorBoard mỗi bước
            # writer.add_scalar('Train/Total_Loss', loss.item(), epoch * len(train_loader) + step)
            # writer.add_scalar('Train/Reconstruction_Loss', R_loss.item(), epoch * len(train_loader) + step)
            # writer.add_scalar('Train/Penalty_Loss', P_loss.item(), epoch * len(train_loader) + step)

            # Cập nhật progress bar
            train_loop.set_postfix({
                "Train_Loss": loss.item(),
            })

        # Log loss trung bình mỗi epoch
        writer.add_scalar('Train/Total_Loss', running_loss / len(train_loader), epoch)

        # validation
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Validate")
        for step, (images, labels) in enumerate(val_loop): 
            with torch.no_grad():
                model.eval()
                images = images.to(device)
                labels = labels.to(device)
                gen_images = model(images, labels)
                val_loss += criterion(gen_images, images)

            val_loop.set_postfix({
                "Val_loss": val_loss.item(),
            })
            
            save_image(images[0].cpu(), os.path.join(log_dir, f"original_batch_{0}.png"))
            save_image(gen_images[0].cpu(), os.path.join(log_dir, f"reconstructed_batch_{0}.png"))
            save_image(images[-1].cpu(), os.path.join(log_dir, f"original_batch_{1}.png"))
            save_image(gen_images[-1].cpu(), os.path.join(log_dir, f"reconstructed_batch_{1}.png"))
            
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best.pt'))  # Lưu mô hình tốt nhất

        writer.add_scalar('Val/Total_Loss', val_loss / len(val_loader), epoch)

        # Lưu mô hình cuối cùng sau mỗi epoch
        torch.save(model.state_dict(), os.path.join(log_dir, 'last.pt'))  # Lưu mô hình cuối cùng


    writer.close()

if __name__ == '__main__':
    train_annotations_file = "dataset/final_malaria_full_class_classification/train_annotation.txt"
    train_img_dir = "dataset/final_malaria_full_class_classification"

    val_annotations_file = "dataset/final_malaria_full_class_classification/val_annotation.txt"
    val_img_dir = "dataset/final_malaria_full_class_classification"

    train_loader = build_dataloader(train_annotations_file,
                                    train_img_dir,
                                    transform, 
                                    batch_size=batch_size, 
                                    minor_classes=[0, 1, 2, 3],
                                    remove_major=True,
                                    num_workers=num_workers,
                                    shuffle=True)

    val_loader = build_dataloader(val_annotations_file,
                                    val_img_dir,
                                    transform=transform,
                                    batch_size=batch_size, 
                                    minor_classes=[0, 1, 2, 3],
                                    remove_major=True,
                                    num_workers=num_workers,
                                    shuffle=False)

    # Initialize models
    train(model, train_loader, val_loader, criterion, optimizer, epochs, device)
    
    