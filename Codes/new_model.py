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
import timm


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
            # Handle triplet batches (anchor, positive, negative, label, metadata)
            if isinstance(b, tuple) and len(b) == 5:
                anchor, positive, negative, labels, metadata = b
                yield to_device(anchor, self.device), to_device(positive, self.device), \
                      to_device(negative, self.device), to_device(labels, self.device), metadata
            else:
                yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()
print(f"Using device: {device}")

# Image transformations with data augmentation
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats for normalization

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Larger size for better feature extraction
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])


class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Create a dictionary of images for each class
        self.class_to_images = {i: [] for i in range(len(self.classes))}
        self.images = []
        
        for cls in self.classes:
            class_path = os.path.join(root_dir, cls)
            class_idx = self.class_to_idx[cls]
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.images.append((img_path, class_idx, cls))
                self.class_to_images[class_idx].append((img_path, cls))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anchor_path, anchor_class, anchor_class_name = self.images[idx]
        
        # Get positive example (same class as anchor)
        positive_images = [img for img in self.class_to_images[anchor_class] if img[0] != anchor_path]
        if positive_images:
            positive_path, positive_class_name = random.choice(positive_images)
        else:
            # If no other images in the same class, use the same image
            positive_path, positive_class_name = anchor_path, anchor_class_name
        
        # Get negative example (different class from anchor)
        negative_class = random.choice([c for c in range(len(self.classes)) if c != anchor_class])
        negative_path, negative_class_name = random.choice(self.class_to_images[negative_class])
        
        # Load images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        # Create labels tensor for multi-task learning
        labels = torch.tensor(anchor_class, dtype=torch.long)
        
        # Return metadata for visualization
        metadata = {
            'anchor_class': anchor_class_name,
            'positive_class': positive_class_name,
            'negative_class': negative_class_name,
            'anchor_path': anchor_path,
            'positive_path': positive_path,
            'negative_path': negative_path
        }
        
        return anchor_img, positive_img, negative_img, labels, metadata


class TripletDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        super(TripletDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=self.collate_fn, **kwargs
        )

    def collate_fn(self, batch):
        anchors = []
        positives = []
        negatives = []
        labels = []
        metadata = []
        
        for anchor, positive, negative, label, meta in batch:
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
            labels.append(label)
            metadata.append(meta)
        
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives), \
               torch.stack(labels), metadata


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        
        spatial_att = self.spatial_attention(x_channel)
        x_out = x_channel * spatial_att
        
        return x_out


class MultiTaskTripletBase(nn.Module):
    def calculate_triplet_loss(self, anchor, positive, negative, margin=0.5):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + margin)
        return losses.mean()
    
    def calculate_classification_loss(self, class_preds, class_labels):
        return F.cross_entropy(class_preds, class_labels)
    
    def training_step(self, batch, alpha=0.7):
        anchor, positive, negative, class_labels, _ = batch
        
        # Get embeddings and class predictions
        anchor_embed, anchor_class = self(anchor)
        positive_embed, _ = self(positive)
        negative_embed, _ = self(negative)
        
        # Calculate losses
        triplet_loss = self.calculate_triplet_loss(anchor_embed, positive_embed, negative_embed)
        classification_loss = self.calculate_classification_loss(anchor_class, class_labels)
        
        # Combined loss
        total_loss = alpha * triplet_loss + (1 - alpha) * classification_loss
        
        return total_loss, triplet_loss, classification_loss
    
    def validation_step(self, batch):
        anchor, positive, negative, class_labels, _ = batch
        
        # Get embeddings and class predictions
        anchor_embed, anchor_class = self(anchor)
        positive_embed, _ = self(positive)
        negative_embed, _ = self(negative)
        
        # Calculate losses
        triplet_loss = self.calculate_triplet_loss(anchor_embed, positive_embed, negative_embed)
        classification_loss = self.calculate_classification_loss(anchor_class, class_labels)
        
        # Calculate accuracies
        _, predicted = anchor_class.max(1)
        correct = predicted.eq(class_labels).sum().item()
        class_acc = correct / len(class_labels)
        
        # Calculate triplet accuracy (positive closer than negative)
        dist_pos = F.pairwise_distance(anchor_embed, positive_embed)
        dist_neg = F.pairwise_distance(anchor_embed, negative_embed)
        triplet_correct = (dist_pos < dist_neg).sum().item()
        triplet_acc = triplet_correct / len(dist_pos)
        
        return {
            'val_triplet_loss': triplet_loss.detach(),
            'val_class_loss': classification_loss.detach(),
            'val_class_acc': torch.tensor(class_acc, device=triplet_loss.device),
            'val_triplet_acc': torch.tensor(triplet_acc, device=triplet_loss.device)
        }
    
    def validation_epoch_end(self, outputs):
        batch_triplet_losses = [x['val_triplet_loss'] for x in outputs]
        batch_class_losses = [x['val_class_loss'] for x in outputs]
        batch_class_accs = [x['val_class_acc'] for x in outputs]
        batch_triplet_accs = [x['val_triplet_acc'] for x in outputs]
        
        epoch_triplet_loss = torch.stack(batch_triplet_losses).mean()
        epoch_class_loss = torch.stack(batch_class_losses).mean()
        epoch_class_acc = torch.stack(batch_class_accs).mean()
        epoch_triplet_acc = torch.stack(batch_triplet_accs).mean()
        
        return {
            'val_triplet_loss': epoch_triplet_loss.item(),
            'val_class_loss': epoch_class_loss.item(),
            'val_class_acc': epoch_class_acc.item(),
            'val_triplet_acc': epoch_triplet_acc.item()
        }
    
    def epoch_end(self, epoch, result, train_losses=None):
        if train_losses:
            print(f"Epoch {epoch}:")
            print(f"  Train total_loss: {train_losses['total']:.4f}, "
                  f"triplet_loss: {train_losses['triplet']:.4f}, "
                  f"class_loss: {train_losses['class']:.4f}")
        
        print(f"  Val triplet_loss: {result['val_triplet_loss']:.4f}, "
              f"class_loss: {result['val_class_loss']:.4f}, "
              f"class_acc: {result['val_class_acc']:.4f}, "
              f"triplet_acc: {result['val_triplet_acc']:.4f}")


class EnhancedTripletNetwork(MultiTaskTripletBase):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        
        # Use face-specific pre-trained model (ResNet50 with IR blocks)
        self.face_backbone = timm.create_model('resnet50', pretrained=True)
        
        # Remove the final classification layer
        self.feature_dim = self.face_backbone.fc.in_features
        self.face_backbone.fc = nn.Identity()
        
        # Attention modules
        self.attention = AttentionModule(2048)
        
        # Embedding network with attention
        self.embedding_network = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Classification head for multi-task learning
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Freeze early layers for transfer learning
        for name, param in self.face_backbone.named_parameters():
            if 'layer4' not in name and 'layer3' not in name:
                param.requires_grad = False
    
    def forward_one(self, x):
        """Forward pass for one input with attention"""
        # Extract features from backbone
        features = self.face_backbone(x)
        
        # If features are 4D (B, C, H, W), apply attention
        if len(features.shape) == 4:
            features = self.attention(features)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        return features
    
    def forward(self, x):
        """Forward pass for the triplet network with multi-task learning"""
        # Extract features
        features = self.forward_one(x)
        
        # Get embeddings
        embeddings = self.embedding_network(features)
        
        # Get class predictions
        class_predictions = self.classifier(embeddings)
        
        return embeddings, class_predictions


class EnsembleTripletNetwork(MultiTaskTripletBase):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        
        # Multiple backbones for ensemble
        self.backbone1 = timm.create_model('resnet50', pretrained=True)
        self.backbone2 = timm.create_model('efficientnet_b0', pretrained=True)
        self.backbone3 = timm.create_model('mobilenetv3_large_100', pretrained=True)
        
        # Get feature dimensions
        self.dim1 = self.backbone1.fc.in_features
        self.dim2 = self.backbone2.classifier.in_features
        self.dim3 = self.backbone3.classifier.in_features
        
        # Remove final layers
        self.backbone1.fc = nn.Identity()
        self.backbone2.classifier = nn.Identity()
        self.backbone3.classifier = nn.Identity()
        
        # Attention modules for each backbone
        self.attention1 = AttentionModule(self.dim1)
        self.attention2 = AttentionModule(self.dim2)
        self.attention3 = AttentionModule(self.dim3)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.dim1 + self.dim2 + self.dim3, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5)
        )
        
        # Embedding network
        self.embedding_network = nn.Sequential(
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features from all backbones
        feat1 = self.backbone1(x)
        feat2 = self.backbone2(x)
        feat3 = self.backbone3(x)
        
        # Apply attention if features are 4D
        if len(feat1.shape) == 4:
            feat1 = self.attention1(feat1)
            feat1 = F.adaptive_avg_pool2d(feat1, (1, 1))
            feat1 = feat1.view(feat1.size(0), -1)
        
        if len(feat2.shape) == 4:
            feat2 = self.attention2(feat2)
            feat2 = F.adaptive_avg_pool2d(feat2, (1, 1))
            feat2 = feat2.view(feat2.size(0), -1)
        
        if len(feat3.shape) == 4:
            feat3 = self.attention3(feat3)
            feat3 = F.adaptive_avg_pool2d(feat3, (1, 1))
            feat3 = feat3.view(feat3.size(0), -1)
        
        # Concatenate features
        fused_features = torch.cat([feat1, feat2, feat3], dim=1)
        fused_features = self.fusion(fused_features)
        
        # Get embeddings and predictions
        embeddings = self.embedding_network(fused_features)
        class_predictions = self.classifier(embeddings)
        
        return embeddings, class_predictions


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
    train_triplet_losses = [x.get('train_triplet_loss', 0) for x in history if 'train_triplet_loss' in x]
    train_class_losses = [x.get('train_class_loss', 0) for x in history if 'train_class_loss' in x]
    val_triplet_losses = [x['val_triplet_loss'] for x in history]
    val_class_losses = [x['val_class_loss'] for x in history]
    val_class_accs = [x['val_class_acc'] for x in history]
    val_triplet_accs = [x['val_triplet_acc'] for x in history]

    # Create figure with 2x2 subplots
    plt.figure(figsize=(15, 10))

    # Plot triplet loss
    plt.subplot(2, 2, 1)
    plt.plot(train_triplet_losses, label='Train Triplet Loss')
    plt.plot(val_triplet_losses, label='Val Triplet Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Triplet Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot classification loss
    plt.subplot(2, 2, 2)
    plt.plot(train_class_losses, label='Train Class Loss')
    plt.plot(val_class_losses, label='Val Class Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Classification Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot classification accuracy
    plt.subplot(2, 2, 3)
    plt.plot(val_class_accs, label='Val Class Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot triplet accuracy
    plt.subplot(2, 2, 4)
    plt.plot(val_triplet_accs, label='Val Triplet Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Triplet Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('triplet_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


# Training function with one-cycle learning rate schedule
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up optimizer
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
        train_total_losses = []
        train_triplet_losses = []
        train_class_losses = []
        
        # Create progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        for batch in loop:
            total_loss, triplet_loss, class_loss = model.training_step(batch)
            
            train_total_losses.append(total_loss.item())
            train_triplet_losses.append(triplet_loss.item())
            train_class_losses.append(class_loss.item())
            
            total_loss.backward()
            
            # Gradient clipping if specified
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
            
            # Update progress bar
            loop.set_postfix(total_loss=total_loss.item(), triplet_loss=triplet_loss.item(), 
                            class_loss=class_loss.item())
        
        # Validation phase
        result = evaluate(model, val_loader)
        
        # Record training losses
        result['train_total_loss'] = np.mean(train_total_losses)
        result['train_triplet_loss'] = np.mean(train_triplet_losses)
        result['train_class_loss'] = np.mean(train_class_losses)
        
        # Record learning rates for this epoch
        result['lrs'] = lrs
        
        # Print epoch results
        model.epoch_end(epoch, result, {
            'total': result['train_total_loss'],
            'triplet': result['train_triplet_loss'],
            'class': result['train_class_loss']
        })
        
        # Save history
        history.append(result)
        
        # Save model after each epoch
        torch.save(model.state_dict(), 'latest_triplet_model.pth')
        
        # Save best model if validation metrics improve
        if epoch == 0 or result['val_class_acc'] > max([h['val_class_acc'] for h in history[:-1]]):
            torch.save(model.state_dict(), 'best_triplet_model.pth')
            print(f"Model saved at epoch {epoch} with val_class_acc: {result['val_class_acc']:.4f}")
    
    return history


# Function to evaluate the model on the test set
def evaluate_triplet_test_set(model, test_dl, classes):
    model.eval()
    all_embeddings = []
    all_class_preds = []
    all_class_labels = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating test set"):
            anchor, positive, negative, class_labels, metadata = batch
            anchor_embed, anchor_class = model(anchor)
            
            all_embeddings.append(anchor_embed.cpu().numpy())
            all_class_preds.append(anchor_class.cpu().numpy())
            all_class_labels.append(class_labels.cpu().numpy())
            all_metadata.extend(metadata)
    
    all_embeddings = np.concatenate(all_embeddings)
    all_class_preds = np.concatenate(all_class_preds)
    all_class_labels = np.concatenate(all_class_labels)
    
    # Get predicted classes
    all_predicted_classes = np.argmax(all_class_preds, axis=1)
    
    # Calculate classification accuracy
    class_acc = accuracy_score(all_class_labels, all_predicted_classes)
    print(f'Test Classification Accuracy: {class_acc * 100:.2f}%')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_class_labels, all_predicted_classes, target_names=classes))
    
    # Generate confusion matrix
    cm = confusion_matrix(all_class_labels, all_predicted_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('triplet_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create embedding visualization using t-SNE
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(12, 10))
    for i, class_name in enumerate(classes):
        mask = all_class_labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   label=class_name, alpha=0.7)
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Embeddings')
    plt.legend()
    plt.tight_layout()
    plt.savefig('triplet_embeddings_tsne.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return class_acc, cm


# Function to visualize triplet predictions
def visualize_triplet_predictions(model, test_dl, classes, num_samples=5):
    # Get a batch of triplets
    dataiter = iter(test_dl)
    anchor, positive, negative, class_labels, metadata = next(dataiter)
    
    # Get embeddings
    model.eval()
    with torch.no_grad():
        anchor_embed, anchor_class = model(anchor)
        positive_embed, _ = model(positive)
        negative_embed, _ = model(negative)
    
    # Convert to CPU
    anchor = anchor.cpu()
    positive = positive.cpu()
    negative = negative.cpu()
    anchor_embed = anchor_embed.cpu()
    positive_embed = positive_embed.cpu()
    negative_embed = negative_embed.cpu()
    anchor_class = anchor_class.cpu()
    
    # Function to denormalize images for display
    def denormalize(image, mean=stats[0], std=stats[1]):
        img_denorm = image.clone()
        for i in range(3):
            img_denorm[i] = img_denorm[i] * std[i] + mean[i]
        return torch.clamp(img_denorm, 0, 1)
    
    # Plot the triplets with distances
    plt.figure(figsize=(15, 12))
    for i in range(min(num_samples, len(anchor))):
        # Denormalize images
        anchor_display = denormalize(anchor[i])
        positive_display = denormalize(positive[i])
        negative_display = denormalize(negative[i])
        
        # Convert to numpy for matplotlib
        anchor_display = anchor_display.permute(1, 2, 0).numpy()
        positive_display = positive_display.permute(1, 2, 0).numpy()
        negative_display = negative_display.permute(1, 2, 0).numpy()
        
        # Plot anchor
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(anchor_display)
        plt.axis('off')
        plt.title(f"Anchor: {metadata[i]['anchor_class']}")
        
        # Plot positive
        plt.subplot(3, num_samples, i + 1 + num_samples)
        plt.imshow(positive_display)
        plt.axis('off')
        plt.title(f"Positive: {metadata[i]['positive_class']}")
        
        # Plot negative
        plt.subplot(3, num_samples, i + 1 + 2*num_samples)
        plt.imshow(negative_display)
        plt.axis('off')
        plt.title(f"Negative: {metadata[i]['negative_class']}")
        
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor_embed[i:i+1], positive_embed[i:i+1])
        neg_dist = F.pairwise_distance(anchor_embed[i:i+1], negative_embed[i:i+1])
        
        # Get class prediction
        _, pred_class = anchor_class[i].max(0)
        pred_class_name = classes[pred_class.item()]
        
        # Add distance information
        color = 'green' if pos_dist < neg_dist else 'red'
        plt.figtext(0.1 + (i * 0.2), 0.01,
                   f"Pos dist: {pos_dist.item():.2f}\n"
                   f"Neg dist: {neg_dist.item():.2f}\n"
                   f"Predicted: {pred_class_name}",
                   color=color, ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig('triplet_sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Create triplet datasets
    train_ds = TripletDataset(train_dir, train_transforms)
    test_ds = TripletDataset(test_dir, test_transforms)
    
    # Create data loaders
    batch_size = 32  # Reduced batch size due to triplet inputs
    train_dl = TripletDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = TripletDataLoader(test_ds, batch_size=batch_size, num_workers=0)
    
    # Move data to device
    train_dl = DeviceDataLoader(train_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)
    
    # Create model (choose between enhanced or ensemble)
    # model = EnhancedTripletNetwork(num_classes=len(classes))
    model = EnsembleTripletNetwork(num_classes=len(classes))
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
        epochs=50,  # Increased epochs for triplet learning
        max_lr=0.001,
        model=model,
        train_loader=train_dl,
        val_loader=test_dl,
        weight_decay=0.001,
        grad_clip=0.1,
        opt_func=torch.optim.Adam
    )
    
    # Plot training metrics
    plot_metrics(history)
    
    # Save the trained model
    torch.save(model.state_dict(), 'final_triplet_model.pth')
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_triplet_model.pth'))
    model = to_device(model, device)
    
    # Evaluate the model on the test dataset
    test_acc, conf_matrix = evaluate_triplet_test_set(model, test_dl, classes)
    
    # Visualize some predictions
    visualize_triplet_predictions(model, test_dl, classes, num_samples=5)
    
    # Plot learning rate vs. loss
    if len(history) > 0 and 'lrs' in history[0]:
        # Extract learning rates and losses
        epochs_lrs = []
        epochs_losses = []
        
        for epoch_result in history:
            if 'lrs' in epoch_result and 'train_total_loss' in epoch_result:
                # We'll use the average LR for the epoch
                avg_lr = sum(epoch_result['lrs']) / len(epoch_result['lrs'])
                epochs_lrs.append(avg_lr)
                epochs_losses.append(epoch_result['train_total_loss'])
        
        # Plot LR vs. Loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_lrs, epochs_losses, 'o-')
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Training Loss')
        plt.title('Learning Rate vs. Training Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('triplet_lr_vs_loss.png', dpi=300, bbox_inches='tight')
        plt.show()


# Additional utility functions for advanced visualization and analysis
def create_embedding_space_visualization(model, data_loader, classes, num_samples=1000):
    """Creates a 3D visualization of the embedding space"""
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i * data_loader.batch_size > num_samples:
                break
            
            anchor, _, _, class_labels, _ = batch
            anchor_embed, _ = model(anchor)
            
            embeddings.append(anchor_embed.cpu().numpy())
            labels.append(class_labels.cpu().numpy())
    
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    # Apply dimensionality reduction
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # PCA for 3D visualization
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, class_name in enumerate(classes):
        mask = labels == i
        ax.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2], 
                  label=class_name, alpha=0.7)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('3D Visualization of Embedding Space')
    ax.legend()
    plt.tight_layout()
    plt.savefig('embedding_space_3d.png', dpi=300, bbox_inches='tight')
    plt.show()


def hard_triplet_mining(anchor_embeddings, positive_embeddings, negative_embeddings):
    """Implements hard triplet mining strategy"""
    # Calculate pairwise distances
    pos_distances = F.pairwise_distance(anchor_embeddings, positive_embeddings)
    neg_distances = F.pairwise_distance(anchor_embeddings, negative_embeddings)
    
    # Find hard positives and hard negatives
    hard_positives = torch.argmax(pos_distances)
    hard_negatives = torch.argmin(neg_distances)
    
    return hard_positives, hard_negatives


def analyze_model_robustness(model, test_loader, classes, noise_levels=[0.1, 0.2, 0.3]):
    """Analyze model robustness to input perturbations"""
    model.eval()
    results = {}
    
    for noise_level in noise_levels:
        accuracies = []
        
        with torch.no_grad():
            for batch in test_loader:
                anchor, _, _, class_labels, _ = batch
                
                # Add Gaussian noise
                noise = torch.randn_like(anchor) * noise_level
                noisy_anchor = anchor + noise
                
                # Get predictions
                _, anchor_class = model(noisy_anchor)
                _, predicted = anchor_class.max(1)
                
                # Calculate accuracy
                correct = predicted.eq(class_labels).sum().item()
                accuracy = correct / len(class_labels)
                accuracies.append(accuracy)
        
        avg_accuracy = np.mean(accuracies)
        results[noise_level] = avg_accuracy
    
    # Plot robustness analysis
    plt.figure(figsize=(10, 6))
    noise_levels_plot = list(results.keys())
    accuracies_plot = list(results.values())
    
    plt.plot(noise_levels_plot, accuracies_plot, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Noise Level')
    plt.ylabel('Classification Accuracy')
    plt.title('Model Robustness to Input Noise')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('model_robustness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    # This is the key to fixing the multiprocessing error
    multiprocessing.freeze_support()
    main()