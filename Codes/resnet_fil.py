import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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

# Load datasets
train_ds = ImageFolder(train_dir, train_transforms)
test_ds = ImageFolder(test_dir, test_transforms)

# Create data loaders
batch_size = 128
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size * 2, num_workers=4, pin_memory=True)

# Move data to device
train_dl = DeviceDataLoader(train_dl, device)
test_dl = DeviceDataLoader(test_dl, device)


# Base model class definition
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result, train_loss=None):
        if train_loss:
            print(
                f"Epoch {epoch}: train_loss: {train_loss:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")
        else:
            print(f"Epoch {epoch}: val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")


# Accuracy calculation function
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# ResNet50 model definition - replacing ResNet152 with ResNet50
class ResNet50Model(ImageClassificationBase):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained ResNet50 model
        self.network = models.resnet50(pretrained=pretrained)

        # Freeze early layers
        for param in list(self.network.parameters())[:-30]:  # Freeze all except last 30 parameter groups
            param.requires_grad = False

        # Replace last layer for our classification task
        in_features = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        return self.network(xb)


# Evaluation function
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
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
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
        torch.save(model.state_dict(), 'latest_model.pth')

        # Save best model if validation accuracy improves
        if epoch == 0 or result['val_acc'] > max([h['val_acc'] for h in history[:-1]]):
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved at epoch {epoch} with val_acc: {result['val_acc']:.4f}")

    return history


# Enhanced test set evaluation function with plotting
def evaluate_test_set(model, test_dl, classes):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating test set"):
            images, labels = batch
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)  # Get the predicted class
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {acc * 100:.2f}%')

    # Print classification report
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return acc, cm


# Create model with proper number of classes
num_classes = len(classes)
model = ResNet50Model(num_classes, pretrained=True)
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
    epochs=50,  # Reduced from 100 to 50 for faster training
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
torch.save(model.state_dict(), 'final_model.pth')

# Load the best model for evaluation
model.load_state_dict(torch.load('best_model.pth'))
model = to_device(model, device)

# Evaluate the model on the test dataset
test_acc, conf_matrix = evaluate_test_set(model, test_dl, classes)


# Function to visualize sample predictions
def visualize_predictions(model, test_dl, classes, num_samples=5):
    # Get a batch of images
    dataiter = iter(test_dl)
    images, labels = next(dataiter)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # Convert tensors to CPU
    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    # Function to denormalize images for display
    def denormalize(image, mean=stats[0], std=stats[1]):
        img_denorm = image.clone()
        for i in range(3):
            img_denorm[i] = img_denorm[i] * std[i] + mean[i]
        return torch.clamp(img_denorm, 0, 1)

    # Plot the images with predictions
    plt.figure(figsize=(15, 3))
    for i in range(min(num_samples, len(images))):
        # Denormalize image
        img = denormalize(images[i])
        # Convert to numpy for matplotlib
        img = img.permute(1, 2, 0).numpy()

        # Plot with colored text based on correctness
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.axis('off')

        # Color text based on correct/incorrect prediction
        true_label = classes[labels[i]]
        pred_label = classes[preds[i]]
        color = 'green' if labels[i] == preds[i] else 'red'

        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)

    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


# Visualize some predictions
visualize_predictions(model, test_dl, classes, num_samples=5)

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
    plt.savefig('lr_vs_loss.png', dpi=300, bbox_inches='tight')
    plt.show()