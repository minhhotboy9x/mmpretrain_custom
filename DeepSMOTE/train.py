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
from module import DeepSMOTE
from sklearn.neighbors import NearestNeighbors
from dataloader import build_dataloader

# ----------------- Hyperparameters ----------------- #
torch.manual_seed(0)
torch.cuda.manual_seed(0)

train_annotations_file = "dataset/final_malaria_full_class_classification/train_annotation.txt"
train_img_dir = "dataset/final_malaria_full_class_classification"

val_annotations_file = "dataset/final_malaria_full_class_classification/val_annotation.txt"
val_img_dir = "dataset/final_malaria_full_class_classification"

batch_size = 4
epochs = 100
lr0 = 1e-3
lrf = 1e-2
momentum = 0.937 # (float) SGD momentum/Adam beta1
weight_decay = 0.0005 # (float) optimizer weight decay 5e-4
num_workers = 4
args = {
        'dim_h': 64,
        'n_channel': 3,
        'n_z': 256
    } 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Thay đổi kích thước ảnh
    transforms.RandomHorizontalFlip(p=0.5), # Lật ngang với xác suất 50%
    transforms.RandomVerticalFlip(p=0.5),   # Lật dọc với xác suất 50%
    transforms.ToTensor()                   # Chuyển ảnh thành Tensor
])

  
# ----------------- DataLoader ----------------- #
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

# ----------------- Criterion ----------------- #
criterion = nn.MSELoss()

# ----------------- Model ----------------- #
model = DeepSMOTE(args)

# ----------------- Optimizer ----------------- #

optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=momentum, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr0*lrf)

# ----------------- Training ----------------- #
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, log_dir="work_dirs/deepsmote_image_logs"):
    
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
        running_P_loss = 0.0
        
        # Vòng lặp huấn luyện
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for step, (images, labels) in enumerate(train_loop):
            images, labels = images.to(device), labels.to(device)
            
            # Sử dụng AMP để huấn luyện
            with amp.autocast(device_type='cuda'): 
                # Reconstruction loss
                outputs = model(images)
                R_loss = criterion(outputs, images)
                
                # Penalty loss
                minor_images, minor_labels = train_loader.dataset.sample_minor_img(train_loader.batch_size)
                minor_images = minor_images.to(device)  # Đưa minor_images lên thiết bị
                e_minor = model.encode(minor_images)
                shuffled_indices = torch.randperm(e_minor.size(0))  # Sinh thứ tự ngẫu nhiên cho b
                e_shuffled_minor = e_minor[shuffled_indices]
                d_shuffled_minor = model.decode(e_shuffled_minor)
                P_loss = criterion(d_shuffled_minor, minor_images)

                # Total loss
                loss = R_loss + P_loss
            
            # Backward và cập nhật optimizer
            optimizer.zero_grad()
            
            # Sử dụng GradScaler để tính toán và cập nhật gradient
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Cộng dồn loss để tính trung bình
            running_loss += loss.item()
            running_R_loss += R_loss.item()
            running_P_loss += P_loss.item()
            
            # Log loss vào TensorBoard mỗi bước
            # writer.add_scalar('Train/Total_Loss', loss.item(), epoch * len(train_loader) + step)
            # writer.add_scalar('Train/Reconstruction_Loss', R_loss.item(), epoch * len(train_loader) + step)
            # writer.add_scalar('Train/Penalty_Loss', P_loss.item(), epoch * len(train_loader) + step)

            # Cập nhật progress bar
            train_loop.set_postfix({
                "Total_Loss": loss.item(),
                "R_Loss": R_loss.item(),
                "P_Loss": P_loss.item()
            })

        # Log loss trung bình mỗi epoch
        writer.add_scalar('Epoch/Total_Loss', running_loss / len(train_loader), epoch)
        writer.add_scalar('Epoch/Reconstruction_Loss', running_R_loss / len(train_loader), epoch)
        writer.add_scalar('Epoch/Penalty_Loss', running_P_loss / len(train_loader), epoch)

        # validation
        val_loss = 0.0
        running_val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Validate")
        for step, (images, labels) in enumerate(val_loop): 
            with torch.no_grad():
                model.eval()
                images = images.to(device)
                gen_images = model(images)
                val_loss = criterion(gen_images, images)
                running_val_loss += val_loss.item()

            val_loop.set_postfix({
                "Val_loss": val_loss.item(),
            })

            
            save_image(images[0].cpu(), os.path.join(log_dir, f"original_batch_{0}.png"))
            save_image(gen_images[0].cpu(), os.path.join(log_dir, f"reconstructed_batch_{0}.png"))
            save_image(images[-1].cpu(), os.path.join(log_dir, f"original_batch_{1}.png"))
            save_image(gen_images[-1].cpu(), os.path.join(log_dir, f"reconstructed_batch_{1}.png"))

        writer.add_scalar('Val/Total_Loss', running_val_loss / len(val_loader), epoch)

        if running_val_loss < best_loss:
            best_loss = running_val_loss
            torch.save({'model': model.state_dict(),
                        'args': model.args,
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'epoch': epoch,
                        'loss': running_val_loss},
                        os.path.join(log_dir, 'best.pt'))  # Lưu mô hình tốt nhất  # Lưu mô hình tốt nhất

        # Lưu mô hình cuối cùng sau mỗi epoch
        torch.save({'model': model.state_dict(),
                    'args': model.args,
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'loss': running_val_loss},
                    os.path.join(log_dir, 'last.pt'))  # Lưu mô hình cuối cùng


    writer.close()

if __name__ == '__main__':
    # train_annotations_file = "dataset/final_malaria_full_class_classification/train_annotation.txt"
    # train_img_dir = "dataset/final_malaria_full_class_classification"

    # val_annotations_file = "dataset/final_malaria_full_class_classification/val_annotation.txt"
    # val_img_dir = "dataset/final_malaria_full_class_classification"

    # train_loader = build_dataloader(train_annotations_file,
    #                             train_img_dir,
    #                             transform, 
    #                             batch_size=batch_size, 
    #                             minor_classes=[0, 1, 2, 3], 
    #                             num_workers=num_workers)

    # val_loader = build_dataloader(val_annotations_file,
    #                                 val_img_dir,
    #                                 transform=transform,
    #                                 batch_size=batch_size, 
    #                                 minor_classes=[0, 1, 2, 3],
    #                                 num_workers=num_workers,
    #                                 shuffle=False)

    # Initialize models
    train(model, train_loader, val_loader, criterion, optimizer, epochs, device)
    
    