# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torchvision import models, transforms
# from torchvision.datasets import ImageFolder
# from tqdm import tqdm
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# import random
# from PIL import Image
# import multiprocessing
#
# # Set seeds for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)
#
# # Dataset paths
# dataset = 'TIF_DF'
# print(os.listdir(dataset))
#
# classes = os.listdir(dataset + "/train")
# print(classes)
#
# data_dir = dataset
# train_dir = data_dir + '/train'
# test_dir = data_dir + '/test'
#
# # Print class distribution
# count = []
# for folder in classes:
#     num_images_train = len(os.listdir(train_dir + '/' + folder))
#     num_images_test = len(os.listdir(test_dir + '/' + folder))
#     count.append(num_images_train)
#     print(f'Training Set: {folder} = {num_images_train}')
#     print(f'Testing Set: {folder} = {num_images_test}')
#     print("--" * 10)
#
#
# # Define device (GPU or CPU)
# def get_default_device():
#     """Pick GPU if available, else CPU"""
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     else:
#         return torch.device('cpu')
#
#
# def to_device(data, device):
#     """Move tensor(s) to chosen device"""
#     if isinstance(data, (list, tuple)):
#         return [to_device(x, device) for x in data]
#     return data.to(device, non_blocking=True)
#
#
# class DeviceDataLoader():
#     """Wrap a dataloader to move data to a device"""
#
#     def __init__(self, dl, device):
#         self.dl = dl
#         self.device = device
#
#     def __iter__(self):
#         """Yield a batch of data after moving it to device"""
#         for b in self.dl:
#             yield to_device(b, self.device)
#
#     def __len__(self):
#         """Number of batches"""
#         return len(self.dl)
#
#
# device = get_default_device()
# print(f"Using device: {device}")
#
# # Image transformations with data augmentation
# stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats for normalization
#
# train_transforms = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(*stats)
# ])
#
# test_transforms = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize(*stats)
# ])
#
#
# # Custom Siamese Dataset
# class SiameseDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = os.listdir(root_dir)
#         self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
#
#         # Create a list of all image paths and their labels
#         self.images = []
#         for cls in self.classes:
#             class_path = os.path.join(root_dir, cls)
#             class_idx = self.class_to_idx[cls]
#             for img_name in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_name)
#                 self.images.append((img_path, class_idx))
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         img1_path, img1_class = self.images[idx]
#
#         # Decide whether to create a similar (same class) or dissimilar (different class) pair
#         should_get_same_class = random.random() < 0.5
#
#         if should_get_same_class:
#             # Find all images of the same class
#             same_class_images = [(p, c) for p, c in self.images if c == img1_class and p != img1_path]
#             if same_class_images:
#                 img2_path, img2_class = random.choice(same_class_images)
#             else:
#                 # If no other images in the same class, use the same image
#                 img2_path, img2_class = img1_path, img1_class
#
#             # Label 1 indicates same class
#             pair_label = torch.tensor(1.0, dtype=torch.float)
#         else:
#             # Find all images of different classes
#             diff_class_images = [(p, c) for p, c in self.images if c != img1_class]
#             img2_path, img2_class = random.choice(diff_class_images)
#
#             # Label 0 indicates different class
#             pair_label = torch.tensor(0.0, dtype=torch.float)
#
#         # Load images
#         img1 = Image.open(img1_path).convert('RGB')
#         img2 = Image.open(img2_path).convert('RGB')
#
#         # Apply transforms if available
#         if self.transform:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
#
#         return img1, img2, pair_label
#
#
# # Siamese Network Base Class
# class SiameseBase(nn.Module):
#     def training_step(self, batch):
#         img1, img2, labels = batch
#         output = self(img1, img2)
#         loss = F.binary_cross_entropy(output, labels)
#         return loss
#
#     def validation_step(self, batch):
#         img1, img2, labels = batch
#         output = self(img1, img2)
#         loss = F.binary_cross_entropy(output, labels)
#
#         # Calculate accuracy (threshold at 0.5)
#         predictions = (output > 0.5).float()
#         correct = torch.eq(predictions, labels).sum().item()
#         acc = correct / len(labels)
#
#         return {'val_loss': loss.detach(), 'val_acc': torch.tensor(acc, device=loss.device)}
#
#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
#
#     def epoch_end(self, epoch, result, train_loss=None):
#         if train_loss:
#             print(
#                 f"Epoch {epoch}: train_loss: {train_loss:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")
#         else:
#             print(f"Epoch {epoch}: val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")
#
#
# # Siamese ResNet50 Model
# class SiameseResNet50(SiameseBase):
#     def __init__(self, pretrained=True):
#         super().__init__()
#
#         # Create the feature extractor using ResNet50 without the final FC layer
#         resnet = models.resnet50(pretrained=pretrained)
#         modules = list(resnet.children())[:-1]  # Remove the final FC layer
#         self.feature_extractor = nn.Sequential(*modules)
#
#         # Freeze early layers for transfer learning
#         for param in list(self.feature_extractor.parameters())[:-30]:
#             param.requires_grad = False
#
#         # Layer to compute similarity
#         self.fc = nn.Sequential(
#             nn.Linear(2048, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#
#     def forward_one(self, x):
#         """Forward pass for one input"""
#         x = self.feature_extractor(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         return x
#
#     def forward(self, x1, x2):
#         """Forward pass for the siamese network"""
#         # Extract features from both inputs
#         feat1 = self.forward_one(x1)
#         feat2 = self.forward_one(x2)
#
#         # Compute absolute difference between features
#         diff = torch.abs(feat1 - feat2)
#
#         # Compute similarity score
#         out = self.fc(diff)
#         return out.squeeze()
#
#
# # Evaluation function for validation
# @torch.no_grad()
# def evaluate(model, val_loader):
#     model.eval()
#     outputs = [model.validation_step(batch) for batch in val_loader]
#     return model.validation_epoch_end(outputs)
#
#
# # Get learning rate from optimizer
# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']
#
#
# # Function to plot training and validation metrics
# def plot_metrics(history):
#     train_losses = [x.get('train_loss', 0) for x in history]
#     val_losses = [x['val_loss'] for x in history]
#     val_accs = [x['val_acc'] for x in history]
#
#     # Create figure with 2 subplots
#     plt.figure(figsize=(14, 5))
#
#     # Plot loss
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses[1:], label='Train Loss')  # Skip the first one as it may not have train loss
#     plt.plot(val_losses, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#
#     # Plot accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(val_accs, label='Val Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Validation Accuracy')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#
#     plt.tight_layout()
#     plt.savefig('siamese_training_metrics.png', dpi=300, bbox_inches='tight')
#     plt.show()
#
#
# # Training function with one-cycle learning rate schedule
# def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
#                   weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
#     torch.cuda.empty_cache()  # Clear GPU memory
#     history = []
#
#     # Set up custom optimizer with weight decay
#     optimizer = opt_func(model.parameters(), lr=max_lr, weight_decay=weight_decay)
#
#     # Set up one-cycle learning rate scheduler
#     sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
#                                                 steps_per_epoch=len(train_loader))
#
#     # For tracking learning rates
#     lrs = []
#
#     # Training loop
#     for epoch in range(epochs):
#         # Training Phase
#         model.train()
#         train_losses = []
#
#         # Create progress bar
#         loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
#
#         for batch in loop:
#             loss = model.training_step(batch)
#             train_losses.append(loss)
#             loss.backward()
#
#             # Gradient clipping if specified
#             if grad_clip:
#                 nn.utils.clip_grad_value_(model.parameters(), grad_clip)
#
#             optimizer.step()
#             optimizer.zero_grad()
#
#             # Record & update learning rate
#             lrs.append(get_lr(optimizer))
#             sched.step()
#
#             # Update progress bar
#             loop.set_postfix(loss=loss.item())
#
#         # Validation phase
#         result = evaluate(model, val_loader)
#
#         # Record training loss
#         result['train_loss'] = torch.stack(train_losses).mean().item()
#
#         # Record learning rates for this epoch
#         result['lrs'] = lrs
#
#         # Print epoch results
#         model.epoch_end(epoch, result, result['train_loss'])
#
#         # Save history
#         history.append(result)
#
#         # Save model after each epoch
#         torch.save(model.state_dict(), 'latest_siamese_model.pth')
#
#         # Save best model if validation accuracy improves
#         if epoch == 0 or result['val_acc'] > max([h['val_acc'] for h in history[:-1]]):
#             torch.save(model.state_dict(), 'best_siamese_model.pth')
#             print(f"Model saved at epoch {epoch} with val_acc: {result['val_acc']:.4f}")
#
#     return history
#
#
# # Function to evaluate the model on the test set with ROC curve
# def evaluate_siamese_test_set(model, test_dl):
#     model.eval()
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for batch in tqdm(test_dl, desc="Evaluating test set"):
#             img1, img2, labels = batch
#             outputs = model(img1, img2)
#             all_preds.append(outputs.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())
#
#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)
#
#     # Convert predictions to binary (0 or 1) using 0.5 threshold
#     binary_preds = (all_preds > 0.5).astype(int)
#
#     # Calculate accuracy
#     acc = accuracy_score(all_labels, binary_preds)
#     print(f'Test Accuracy: {acc * 100:.2f}%')
#
#     # Print classification report
#     print(classification_report(all_labels, binary_preds, target_names=['Different', 'Same']))
#
#     # Generate confusion matrix
#     cm = confusion_matrix(all_labels, binary_preds)
#     print("Confusion Matrix:")
#     print(cm)
#
#     # Plot confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=['Different', 'Same'],
#                 yticklabels=['Different', 'Same'])
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#     plt.savefig('siamese_confusion_matrix.png', dpi=300, bbox_inches='tight')
#     plt.show()
#
#     # Calculate ROC curve and AUC
#     from sklearn.metrics import roc_curve, auc
#     fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
#     roc_auc = auc(fpr, tpr)
#
#     # Plot ROC curve
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")
#     plt.savefig('siamese_roc_curve.png', dpi=300, bbox_inches='tight')
#     plt.show()
#
#     return acc, cm, roc_auc
#
#
# # Function to visualize sample pair predictions
# def visualize_pair_predictions(model, test_dl, num_samples=5):
#     # Get a batch of image pairs
#     dataiter = iter(test_dl)
#     img1, img2, labels = next(dataiter)
#
#     # Get predictions
#     model.eval()
#     with torch.no_grad():
#         outputs = model(img1, img2)
#         preds = (outputs > 0.5).float()
#
#     # Convert tensors to CPU
#     img1 = img1.cpu()
#     img2 = img2.cpu()
#     labels = labels.cpu()
#     outputs = outputs.cpu()
#     preds = preds.cpu()
#
#     # Function to denormalize images for display
#     def denormalize(image, mean=stats[0], std=stats[1]):
#         img_denorm = image.clone()
#         for i in range(3):
#             img_denorm[i] = img_denorm[i] * std[i] + mean[i]
#         return torch.clamp(img_denorm, 0, 1)
#
#     # Plot the pairs with predictions
#     plt.figure(figsize=(15, 6))
#     for i in range(min(num_samples, len(img1))):
#         # Denormalize images
#         img1_display = denormalize(img1[i])
#         img2_display = denormalize(img2[i])
#
#         # Convert to numpy for matplotlib
#         img1_display = img1_display.permute(1, 2, 0).numpy()
#         img2_display = img2_display.permute(1, 2, 0).numpy()
#
#         # Plot image pair
#         plt.subplot(2, num_samples, i + 1)
#         plt.imshow(img1_display)
#         plt.axis('off')
#         plt.title("Image 1")
#
#         plt.subplot(2, num_samples, i + 1 + num_samples)
#         plt.imshow(img2_display)
#         plt.axis('off')
#         plt.title("Image 2")
#
#         # Add prediction information
#         true_label = "Same" if labels[i] == 1 else "Different"
#         pred_label = "Same" if preds[i] == 1 else "Different"
#         color = 'green' if labels[i] == preds[i] else 'red'
#         similarity_score = outputs[i].item()
#
#         plt.figtext(0.1 + (i * 0.2), 0.01,
#                     f"True: {true_label}\nPred: {pred_label}\nScore: {similarity_score:.2f}",
#                     color=color, ha='center')
#
#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.2)
#     plt.savefig('siamese_sample_predictions.png', dpi=300, bbox_inches='tight')
#     plt.show()
#
#
# def main():
#     # Required for Windows to avoid the multiprocessing error
#
#     # Create Siamese datasets
#     train_ds = SiameseDataset(train_dir, train_transforms)
#     test_ds = SiameseDataset(test_dir, test_transforms)
#
#     # Create data loaders with num_workers=0 to avoid multiprocessing issues
#     batch_size = 64
#     train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0)
#     test_dl = DataLoader(test_ds, batch_size, num_workers=0)
#
#     # Move data to device
#     train_dl = DeviceDataLoader(train_dl, device)
#     test_dl = DeviceDataLoader(test_dl, device)
#
#     # Create Siamese model
#     model = SiameseResNet50(pretrained=True)
#     model = to_device(model, device)
#
#     # Print model summary
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
#     print(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%")
#
#     # Get initial validation metrics
#     initial_result = evaluate(model, test_dl)
#     print("Initial validation metrics:", initial_result)
#
#     # Train the model
#     history = fit_one_cycle(
#         epochs=30,  # Reduced epochs for siamese network
#         max_lr=0.001,
#         model=model,
#         train_loader=train_dl,
#         val_loader=test_dl,
#         weight_decay=0.01,
#         grad_clip=0.1,
#         opt_func=torch.optim.Adam
#     )
#
#     # Plot training metrics
#     plot_metrics(history)
#
#     # Save the trained model
#     torch.save(model.state_dict(), 'final_siamese_model.pth')
#
#     # Load the best model for evaluation
#     model.load_state_dict(torch.load('best_siamese_model.pth'))
#     model = to_device(model, device)
#
#     # Evaluate the model on the test dataset
#     test_acc, conf_matrix, roc_auc = evaluate_siamese_test_set(model, test_dl)
#
#     # Visualize some predictions
#     visualize_pair_predictions(model, test_dl, num_samples=5)
#
#     # Plot learning rate vs. loss
#     if len(history) > 0 and 'lrs' in history[0]:
#         # Extract learning rates and losses
#         epochs_lrs = []
#         epochs_losses = []
#
#         for epoch_result in history:
#             if 'lrs' in epoch_result and 'train_loss' in epoch_result:
#                 # We'll use the average LR for the epoch
#                 avg_lr = sum(epoch_result['lrs']) / len(epoch_result['lrs'])
#                 epochs_lrs.append(avg_lr)
#                 epochs_losses.append(epoch_result['train_loss'])
#
#         # Plot LR vs. Loss
#         plt.figure(figsize=(10, 6))
#         plt.plot(epochs_lrs, epochs_losses, 'o-')
#         plt.xscale('log')
#         plt.xlabel('Learning Rate (log scale)')
#         plt.ylabel('Training Loss')
#         plt.title('Learning Rate vs. Training Loss')
#         plt.grid(True, linestyle='--', alpha=0.6)
#         plt.savefig('siamese_lr_vs_loss.png', dpi=300, bbox_inches='tight')
#         plt.show()
#
#
# if __name__ == "__main__":
#     # This is the key to fixing the multiprocessing error
#     multiprocessing.freeze_support()
#     main()

# ------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import random
from PIL import Image
import multiprocessing

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Dataset paths
dataset = 'TIF_DF'
print(os.listdir(dataset))

classes = os.listdir(dataset + "/train")
print(classes)

data_dir = dataset
train_dir = data_dir + '/train'
test_dir = data_dir + '/test'

# Print class distribution
count = []
for folder in classes:
    num_images_train = len(os.listdir(train_dir + '/' + folder))
    num_images_test = len(os.listdir(test_dir + '/' + folder))
    count.append(num_images_train)
    print(f'Training Set: {folder} = {num_images_train}')
    print(f'Testing Set: {folder} = {num_images_test}')
    print("--" * 10)


# Define device (GPU or CPU)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            # Check if we have a tuple of 4 elements (img1, img2, labels, metadata)
            if isinstance(b, tuple) and len(b) == 4:
                img1, img2, labels, metadata = b
                # Only move tensors to device, leave metadata as is
                yield to_device(img1, self.device), to_device(img2, self.device), to_device(labels, self.device), metadata
            else:
                # For other types of batches
                yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()
print(f"Using device: {device}")

# Image transformations with data augmentation
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats for normalization

train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_transforms = transforms.Compose([

    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])


# Custom Siamese Dataset that works with emotion classes
class SiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Create a list of all image paths and their labels
        self.images = []
        for cls in self.classes:
            class_path = os.path.join(root_dir, cls)
            class_idx = self.class_to_idx[cls]
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.images.append((img_path, class_idx, cls))  # Store class name as well

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img1_path, img1_class, img1_class_name = self.images[idx]

        # Decide whether to create a similar (same class) or dissimilar (different class) pair
        should_get_same_class = random.random() < 0.5

        if should_get_same_class:
            # Find all images of the same class
            same_class_images = [(p, c, n) for p, c, n in self.images if c == img1_class and p != img1_path]
            if same_class_images:
                img2_path, img2_class, img2_class_name = random.choice(same_class_images)
            else:
                # If no other images in the same class, use the same image
                img2_path, img2_class, img2_class_name = img1_path, img1_class, img1_class_name

            # Label 1 indicates same class
            pair_label = torch.tensor(1.0, dtype=torch.float)
        else:
            # Find all images of different classes
            diff_class_images = [(p, c, n) for p, c, n in self.images if c != img1_class]
            img2_path, img2_class, img2_class_name = random.choice(diff_class_images)

            # Label 0 indicates different class
            pair_label = torch.tensor(0.0, dtype=torch.float)

        # Load images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # Apply transforms if available
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Return metadata for visualization
        metadata = {
            'img1_class': img1_class_name,
            'img2_class': img2_class_name,
            'img1_path': img1_path,
            'img2_path': img2_path
        }

        return img1, img2, pair_label, metadata


# Modified DataLoader to handle metadata
class SiameseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        super(SiameseDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=self.collate_fn, **kwargs
        )

    def collate_fn(self, batch):
        imgs1 = []
        imgs2 = []
        labels = []
        metadata = []

        for img1, img2, label, meta in batch:
            imgs1.append(img1)
            imgs2.append(img2)
            labels.append(label)
            metadata.append(meta)

        return torch.stack(imgs1), torch.stack(imgs2), torch.stack(labels), metadata


# Siamese Network Base Class
class SiameseBase(nn.Module):
    def training_step(self, batch):
        img1, img2, labels, _ = batch  # Ignore metadata for training
        output = self(img1, img2)
        loss = F.binary_cross_entropy(output, labels)
        return loss

    def validation_step(self, batch):
        img1, img2, labels, _ = batch  # Ignore metadata for validation
        output = self(img1, img2)
        loss = F.binary_cross_entropy(output, labels)

        # Calculate accuracy (threshold at 0.5)
        predictions = (output > 0.5).float()
        correct = torch.eq(predictions, labels).sum().item()
        acc = correct / len(labels)

        return {'val_loss': loss.detach(), 'val_acc': torch.tensor(acc, device=loss.device)}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result, train_loss=None):
        if train_loss:
            print(
                f"Epoch {epoch}: train_loss: {train_loss:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")
        else:
            print(f"Epoch {epoch}: val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")


# Siamese ResNet50 Model
class SiameseResNet50(SiameseBase):
    def __init__(self, pretrained=True):
        super().__init__()

        # Create the feature extractor using ResNet50 without the final FC layer
        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-1]  # Remove the final FC layer
        self.feature_extractor = nn.Sequential(*modules)

        # Freeze early layers for transfer learning
        for param in list(self.feature_extractor.parameters())[:-30]:
            param.requires_grad = False

        # Layer to compute similarity
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        """Forward pass for one input"""
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x

    def forward(self, x1, x2):
        """Forward pass for the siamese network"""
        # Extract features from both inputs
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)

        # Compute absolute difference between features
        diff = torch.abs(feat1 - feat2)

        # Compute similarity score
        out = self.fc(diff)
        return out.squeeze()


# Evaluation function for validation
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# Get learning rate from optimizer
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# Function to plot training and validation metrics
def plot_metrics(history):
    train_losses = [x.get('train_loss', 0) for x in history]
    val_losses = [x['val_loss'] for x in history]
    val_accs = [x['val_acc'] for x in history]

    # Create figure with 2 subplots
    plt.figure(figsize=(14, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses[1:], label='Train Loss')  # Skip the first one as it may not have train loss
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('siamese_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


# Training function with one-cycle learning rate schedule
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()  # Clear GPU memory
    history = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), lr=max_lr, weight_decay=weight_decay)

    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    # For tracking learning rates
    lrs = []

    # Training loop
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []

        # Create progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for batch in loop:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping if specified
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

            # Update progress bar
            loop.set_postfix(loss=loss.item())

        # Validation phase
        result = evaluate(model, val_loader)

        # Record training loss
        result['train_loss'] = torch.stack(train_losses).mean().item()

        # Record learning rates for this epoch
        result['lrs'] = lrs

        # Print epoch results
        model.epoch_end(epoch, result, result['train_loss'])

        # Save history
        history.append(result)

        # Save model after each epoch
        torch.save(model.state_dict(), 'latest_siamese_model.pth')

        # Save best model if validation accuracy improves
        if epoch == 0 or result['val_acc'] > max([h['val_acc'] for h in history[:-1]]):
            torch.save(model.state_dict(), 'best_siamese_model.pth')
            print(f"Model saved at epoch {epoch} with val_acc: {result['val_acc']:.4f}")

    return history


# Function to evaluate the model on the test set with emotion classes
def evaluate_siamese_test_set(model, test_dl, classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_metadata = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating test set"):
            img1, img2, labels, metadata = batch
            outputs = model(img1, img2)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_metadata.extend(metadata)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Convert predictions to binary (0 or 1) using 0.5 threshold
    binary_preds = (all_preds > 0.5).astype(int)

    # Calculate accuracy
    acc = accuracy_score(all_labels, binary_preds)
    print(f'Test Accuracy: {acc * 100:.2f}%')

    # Print classification report
    print(classification_report(all_labels, binary_preds, target_names=['Different', 'Same']))

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, binary_preds)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Different', 'Same'],
                yticklabels=['Different', 'Same'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('siamese_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate ROC curve and AUC
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('siamese_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Analyze class-wise performance
    print("\nEmotion class similarity analysis:")

    # Count correct predictions for each class pair
    class_pair_counts = {}
    class_pair_correct = {}

    for i in range(len(all_labels)):
        img1_class = all_metadata[i]['img1_class']
        img2_class = all_metadata[i]['img2_class']
        pair_key = f"{img1_class}-{img2_class}"

        # Initialize if not seen before
        if pair_key not in class_pair_counts:
            class_pair_counts[pair_key] = 0
            class_pair_correct[pair_key] = 0

        class_pair_counts[pair_key] += 1
        if all_labels[i] == binary_preds[i]:
            class_pair_correct[pair_key] += 1

    # Calculate and print accuracy for each class pair
    print("\nPair-wise accuracies:")
    class_pair_acc = {}
    for pair, count in class_pair_counts.items():
        if count > 0:
            acc = class_pair_correct[pair] / count
            class_pair_acc[pair] = acc
            print(f"{pair}: {acc:.4f} ({class_pair_correct[pair]}/{count})")

    # Create a matrix of similarities between classes
    similarity_matrix = np.zeros((len(classes), len(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for i in range(len(all_metadata)):
        if all_labels[i] == 1:  # Only for pairs that should be similar
            cls1 = all_metadata[i]['img1_class']
            cls2 = all_metadata[i]['img2_class']
            similarity = all_preds[i]

            idx1 = class_to_idx[cls1]
            idx2 = class_to_idx[cls2]

            # Update similarity (we'll average it later)
            similarity_matrix[idx1, idx2] += similarity
            similarity_matrix[idx2, idx1] += similarity  # Make it symmetric

    # Average similarities
    class_pair_counts_matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(all_metadata)):
        if all_labels[i] == 1:
            cls1 = all_metadata[i]['img1_class']
            cls2 = all_metadata[i]['img2_class']
            idx1 = class_to_idx[cls1]
            idx2 = class_to_idx[cls2]
            class_pair_counts_matrix[idx1, idx2] += 1
            class_pair_counts_matrix[idx2, idx1] += 1

    # Avoid division by zero
    class_pair_counts_matrix = np.maximum(class_pair_counts_matrix, 1)
    similarity_matrix = similarity_matrix / class_pair_counts_matrix

    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Emotion Class")
    plt.ylabel("Emotion Class")
    plt.title("Emotion Class Similarity Matrix")
    plt.tight_layout()
    plt.savefig('emotion_similarity_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return acc, cm, roc_auc, similarity_matrix


# Function to visualize sample pair predictions with emotion classes
def visualize_pair_predictions(model, test_dl, classes, num_samples=5):
    # Get a batch of image pairs
    dataiter = iter(test_dl)
    img1, img2, labels, metadata = next(dataiter)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(img1, img2)
        preds = (outputs > 0.5).float()

    # Convert tensors to CPU
    img1 = img1.cpu()
    img2 = img2.cpu()
    labels = labels.cpu()
    outputs = outputs.cpu()
    preds = preds.cpu()

    # Function to denormalize images for display
    def denormalize(image, mean=stats[0], std=stats[1]):
        img_denorm = image.clone()
        for i in range(3):
            img_denorm[i] = img_denorm[i] * std[i] + mean[i]
        return torch.clamp(img_denorm, 0, 1)

    # Plot the pairs with predictions
    plt.figure(figsize=(15, 8))
    for i in range(min(num_samples, len(img1))):
        # Denormalize images
        img1_display = denormalize(img1[i])
        img2_display = denormalize(img2[i])

        # Convert to numpy for matplotlib
        img1_display = img1_display.permute(1, 2, 0).numpy()
        img2_display = img2_display.permute(1, 2, 0).numpy()

        # Plot image pair
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(img1_display)
        plt.axis('off')
        plt.title(f"Image 1: {metadata[i]['img1_class']}")

        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(img2_display)
        plt.axis('off')
        plt.title(f"Image 2: {metadata[i]['img2_class']}")

        # Add prediction information
        true_label = "Same" if labels[i] == 1 else "Different"
        pred_label = "Same" if preds[i] == 1 else "Different"
        color = 'green' if labels[i] == preds[i] else 'red'
        similarity_score = outputs[i].item()

        plt.figtext(0.1 + (i * 0.2), 0.01,
                    f"True: {true_label}\nPred: {pred_label}\nScore: {similarity_score:.2f}",
                    color=color, ha='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig('siamese_sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Required for Windows to avoid the multiprocessing error

    # Create Siamese datasets with metadata
    train_ds = SiameseDataset(train_dir, train_transforms)
    test_ds = SiameseDataset(test_dir, test_transforms)

    # Create data loaders with custom collate function
    batch_size = 64
    train_dl = SiameseDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = SiameseDataLoader(test_ds, batch_size=batch_size, num_workers=0)

    # Move data to device
    train_dl = DeviceDataLoader(train_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)

    # Create Siamese model
    model = SiameseResNet50(pretrained=True)
    model = to_device(model, device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%")

    # Get initial validation metrics
    initial_result = evaluate(model, test_dl)
    print("Initial validation metrics:", initial_result)

    # Train the model
    history = fit_one_cycle(
        epochs=30,  # Reduced epochs for siamese network
        max_lr=0.001,
        model=model,
        train_loader=train_dl,
        val_loader=test_dl,
        weight_decay=0.01,
        grad_clip=0.1,
        opt_func=torch.optim.Adam
    )

    # Plot training metrics
    plot_metrics(history)

    # Save the trained model
    torch.save(model.state_dict(), 'final_siamese_model.pth')

    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_siamese_model.pth'))
    model = to_device(model, device)

    # Evaluate the model on the test dataset
    test_acc, conf_matrix, roc_auc, similarity_matrix = evaluate_siamese_test_set(model, test_dl, classes)

    # Visualize some predictions with emotion classes
    visualize_pair_predictions(model, test_dl, classes, num_samples=5)

    # Plot learning rate vs. loss
    if len(history) > 0 and 'lrs' in history[0]:
        # Extract learning rates and losses
        epochs_lrs = []
        epochs_losses = []

        for epoch_result in history:
            if 'lrs' in epoch_result and 'train_loss' in epoch_result:
                # We'll use the average LR for the epoch
                avg_lr = sum(epoch_result['lrs']) / len(epoch_result['lrs'])
                epochs_lrs.append(avg_lr)
                epochs_losses.append(epoch_result['train_loss'])

        # Plot LR vs. Loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_lrs, epochs_losses, 'o-')
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Training Loss')
        plt.title('Learning Rate vs. Training Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('siamese_lr_vs_loss.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # This is the key to fixing the multiprocessing error
    multiprocessing.freeze_support()
    main()