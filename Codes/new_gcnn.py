import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data
from scipy.spatial import Delaunay
import ast
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

# Suppress specific warning by category
warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")


def load_and_preprocess_data(csv_path='tif_df_train.csv', test_size=0.2, random_state=42):
    """
    Load and preprocess the facial landmark data from CSV.

    Args:
        csv_path: Path to the CSV file containing landmark data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        encoder: Label encoder for emotion classes
    """
    print(f"Loading data from {csv_path}...")

    # Load the dataframe
    df = pd.read_csv(csv_path)

    # Get emotion labels and encode them
    emotion_list = ["Angry", "Disgust", "Fear", "Happiness", "Neutral", "Sad", "Surprised"]
    encoder = LabelEncoder()
    df['encoded_label'] = encoder.fit_transform(df['Labels'])

    # Print dataset information
    print(f"Dataset contains {len(df)} samples")
    print(f"Emotion distribution:\n{df['Labels'].value_counts()}")

    # Prepare data
    X = df['Landmarks']
    y = df['encoded_label']

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Create datasets and dataloaders
    train_dataset = LandmarkDataset(X_train, y_train)
    val_dataset = LandmarkDataset(X_val, y_val)

    # Use the PyTorch Geometric DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, encoder


def delaunay_triangulation(landmarks):
    """
    Perform Delaunay triangulation on landmarks.

    Args:
        landmarks: Array of shape (num_landmarks, 2) containing landmark coordinates

    Returns:
        triangles: Delaunay triangles
        edges: List of edges connecting landmarks
    """
    # Create Delaunay triangulation
    try:
        tri = Delaunay(landmarks)
        triangles = tri.simplices

        # Extract edges from triangles
        edges = set()
        for triangle in triangles:
            for i in range(3):
                edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
                edges.add(edge)

        return triangles, list(edges)
    except Exception as e:
        print(f"Error in Delaunay triangulation: {e}")
        # Fall back to a simple circular graph structure
        num_landmarks = len(landmarks)
        edges = [(i, (i + 1) % num_landmarks) for i in range(num_landmarks)]
        return None, edges


def construct_adjacency_matrix(landmarks):
    """
    Construct adjacency matrix from landmarks using Delaunay triangulation.

    Args:
        landmarks: Array of shape (num_landmarks, 2) containing landmark coordinates

    Returns:
        adjacency_matrix: Binary adjacency matrix
    """
    _, edges = delaunay_triangulation(landmarks)
    num_landmarks = len(landmarks)
    adjacency_matrix = np.zeros((num_landmarks, num_landmarks), dtype=int)

    for edge in edges:
        i, j = edge
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  # Symmetric for undirected graph

    # Add self-loops
    np.fill_diagonal(adjacency_matrix, 1)

    return adjacency_matrix


def normalize_landmarks(landmarks):
    """
    Normalize landmark coordinates to [0, 1] range.

    Args:
        landmarks: Array of shape (num_landmarks, 2) containing landmark coordinates

    Returns:
        normalized_landmarks: Normalized landmark coordinates
    """
    # Find min and max for each dimension
    min_coords = np.min(landmarks, axis=0)
    max_coords = np.max(landmarks, axis=0)

    # Avoid division by zero
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1  # Avoid division by zero

    # Normalize to [0, 1]
    normalized_landmarks = (landmarks - min_coords) / range_coords

    return normalized_landmarks


def process_landmarks_and_adjacency(landmarks_str):
    """
    Process landmark string into normalized landmarks and adjacency matrix.

    Args:
        landmarks_str: String representation of landmarks

    Returns:
        landmarks: Normalized landmark coordinates
        adjacency_matrix: Binary adjacency matrix
    """
    try:
        # Parse the string to get the list of landmarks
        landmarks = np.array(ast.literal_eval(landmarks_str))

        # Normalize landmarks
        landmarks = normalize_landmarks(landmarks)

        # Get adjacency matrix
        adjacency_matrix = construct_adjacency_matrix(landmarks)

        return landmarks, adjacency_matrix
    except Exception as e:
        print(f"Error processing landmarks: {e}")
        # Return empty arrays if parsing fails
        return np.zeros((68, 2)), np.zeros((68, 68))


class LandmarkDataset(Dataset):
    """
    Dataset for facial landmarks.
    """

    def __init__(self, X, y):
        """
        Initialize the dataset.

        Args:
            X: Series of landmark strings
            y: Series of emotion labels
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]

        # Process the landmarks and adjacency matrix
        landmarks, adjacency_matrix = process_landmarks_and_adjacency(row)

        # Convert landmarks to tensor (Shape: 68, 2)
        x = torch.tensor(landmarks, dtype=torch.float)

        # Convert adjacency matrix to edge_index (Shape: 2, num_edges)
        edge_index = torch.tensor(np.array(np.nonzero(adjacency_matrix)), dtype=torch.long)

        # Get the label for this sample
        y = torch.tensor(self.y.iloc[idx], dtype=torch.long)  # Label

        return Data(x=x, edge_index=edge_index, y=y)


class GCNModel(nn.Module):
    """
    Graph Convolutional Network model for facial expression recognition.
    """

    def __init__(self, num_nodes=68, num_features=2, num_classes=7, dropout_rate=0.3):
        """
        Initialize the GCN model.

        Args:
            num_nodes: Number of landmark nodes
            num_features: Number of features per node (usually 2 for 2D coordinates)
            num_classes: Number of emotion classes
            dropout_rate: Dropout probability
        """
        super(GCNModel, self).__init__()

        # Convolutional layers
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, 512)
        self.conv5 = GCNConv(512, 1024)

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers after pooling
        self.fc1 = nn.Linear(2048, 512)  # 2048 = 1024 (mean pool) + 1024 (max pool)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x, edge_index, batch):
        """
        Forward pass through the network.

        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch assignment for nodes

        Returns:
            logits: Class prediction logits
        """
        # First convolutional layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Second convolutional layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Third convolutional layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Fourth convolutional layer
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Fifth convolutional layer
        x = self.conv5(x, edge_index)
        x = F.relu(x)

        # Pooling layers - applying both mean and max pooling
        x_mean = global_mean_pool(x, batch)  # Global mean pooling
        x_max = global_max_pool(x, batch)  # Global max pooling

        # Concatenate the pooled results
        x = torch.cat([x_mean, x_max], dim=1)  # Shape: [batch_size, 2048]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def train(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Train the model for one epoch.

    Args:
        model: The GCN model
        loader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on
        scheduler: Learning rate scheduler (optional)

    Returns:
        train_loss: Average training loss
        train_accuracy: Training accuracy
        confusion_mat: Confusion matrix
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    # Use tqdm for progress bar
    pbar = tqdm(loader, desc="Training")

    for batch in pbar:
        batch = batch.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)

        # Compute loss
        loss = criterion(out, batch.y)
        total_loss += loss.item() * batch.num_graphs

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(out, dim=1)
        correct += (predicted == batch.y).sum().item()
        total += batch.y.size(0)

        # Store predictions and labels for confusion matrix
        all_labels.extend(batch.y.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})

    # Step the scheduler if provided
    if scheduler is not None:
        scheduler.step()

    train_loss = total_loss / total
    train_accuracy = correct / total

    # Compute confusion matrix
    confusion_mat = confusion_matrix(all_labels, all_preds)

    return train_loss, train_accuracy, confusion_mat


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model.

    Args:
        model: The GCN model
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        val_loss: Average validation loss
        val_accuracy: Validation accuracy
        confusion_mat: Confusion matrix
        all_preds: All predictions
        all_labels: All true labels
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)

            # Compute loss
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs

            # Calculate accuracy
            _, predicted = torch.max(out, dim=1)
            correct += (predicted == batch.y).sum().item()
            total += batch.y.size(0)

            # Store predictions and labels
            all_labels.extend(batch.y.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_loss = total_loss / total
    val_accuracy = correct / total

    # Compute confusion matrix
    confusion_mat = confusion_matrix(all_labels, all_preds)

    return val_loss, val_accuracy, confusion_mat, all_preds, all_labels


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir=None):
    """
    Plot training and validation metrics.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
        save_dir: Directory to save plots (optional)
    """
    # Create the directory if it doesn't exist
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.show()

    # Plot accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.show()


def plot_confusion_matrix(cm, labels, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        labels: Class labels
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_model(model, filepath):
    """
    Save model state dict.

    Args:
        model: The model to save
        filepath: Path to save the model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device):
    """
    Load model state dict.

    Args:
        model: The model to load weights into
        filepath: Path to the saved model
        device: Device to load the model on

    Returns:
        model: The loaded model
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"Model loaded from {filepath}")
    return model


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create directories for saving results
    save_dir = 'gcnn_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, encoder = load_and_preprocess_data()

    # Get emotion labels
    emotion_labels = encoder.classes_

    # Initialize model
    model = GCNModel(num_nodes=68, num_features=2, num_classes=len(emotion_labels)).to(device)

    # Print model summary
    print(model)

    # Optimizer, Loss and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training and evaluation
    epochs = 50
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss, train_accuracy, cm_train = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validate
        val_loss, val_accuracy, cm_val, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Update scheduler
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, os.path.join(save_dir, 'best_model.pth'))
            print(f"New best validation accuracy: {best_val_accuracy * 100:.2f}%")

            # Save the confusion matrix for the best model
            plot_confusion_matrix(
                cm_val,
                emotion_labels,
                title='Validation Confusion Matrix (Best Model)',
                save_path=os.path.join(save_dir, 'best_confusion_matrix.png')
            )

    # Plot and save metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir)

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=emotion_labels))

    # Save the final model
    save_model(model, os.path.join(save_dir, 'final_model.pth'))

    print("Training complete!")


if __name__ == '__main__':
    main()