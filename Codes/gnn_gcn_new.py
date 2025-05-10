import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings

# Suppress specific warning by category
warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the dataframe
df_train = pd.read_csv('child_train.csv')
df_test = pd.read_csv('child_test.csv')

# Initialize the label encoder
emotion_list = ["Angry", "Disgust", "Fear", "Happiness", "Neutral", "Sad", "Surprised"]
encoder = LabelEncoder()
encoder.fit(emotion_list)
df_train['encoded_label'] = encoder.transform(df_train['Labels'])
df_test['encoded_label'] = encoder.transform(df_test['Labels'])

# Function to construct adjacency matrix using distance-based thresholding
def construct_adjacency_matrix(landmarks, threshold=None):
    """
    Create an adjacency matrix based on Euclidean distances between landmarks.
    Connections are established when distance < threshold.
    
    Args:
        landmarks: Numpy array of shape (num_landmarks, 2) containing landmark coordinates
        threshold: Distance threshold for connecting landmarks. If None, calculated automatically.
    
    Returns:
        adjacency_matrix: Binary matrix where A_ij = 1 if landmarks i and j are connected
        valid_connections: List of tuples (i,j) representing connected landmarks
    """
    num_landmarks = len(landmarks)
    
    # Create adjacency matrix
    adjacency_matrix = np.zeros((num_landmarks, num_landmarks), dtype=int)
    
    # Calculate all pairwise distances between landmarks
    distances = np.zeros((num_landmarks, num_landmarks))
    for i in range(num_landmarks):
        for j in range(i+1, num_landmarks):  # Only calculate upper triangle
            dist = np.linalg.norm(landmarks[i] - landmarks[j])
            distances[i, j] = dist
            distances[j, i] = dist  # Mirror to lower triangle
    
    # If threshold is not provided, calculate it automatically
    # Use a percentile of the distance distribution as the threshold
    if threshold is None:
        # Flatten the upper triangle (excluding diagonal) to get all unique distances
        unique_distances = distances[np.triu_indices(num_landmarks, k=1)]
        # Use a percentile as the threshold (e.g., 20th percentile)
        threshold = np.percentile(unique_distances, 20)  # Can be adjusted for density control
    
    # Apply thresholding: create edges between landmarks whose distance is below threshold
    valid_connections = []
    for i in range(num_landmarks):
        for j in range(i+1, num_landmarks):  # Only process upper triangle
            if distances[i, j] < threshold:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1  # Symmetric for undirected graph
                valid_connections.append((i, j))
    
    # Fallback if no valid connections were found
    if not valid_connections:
        print("Warning: No connections were made using the threshold. Using fallback chain connections.")
        for i in range(num_landmarks):
            next_idx = (i+1) % num_landmarks
            adjacency_matrix[i, next_idx] = 1
            adjacency_matrix[next_idx, i] = 1
            valid_connections.append((i, next_idx))
    
    return adjacency_matrix, valid_connections

# Normalize adjacency matrix
def normalize_adjacency_matrix(A):
    """
    Normalize adjacency matrix using symmetric normalization:
    A_norm = D^(-1/2) * A * D^(-1/2)
    """
    # Add self-loops
    A = A + np.eye(A.shape[0])
    
    # Calculate degree matrix
    D = np.sum(A, axis=1)
    
    # Avoid division by zero
    D_inv_sqrt = 1.0 / np.sqrt(np.maximum(D, 1e-12))
    D_inv_sqrt = np.diag(D_inv_sqrt)
    
    # Normalize using D^(-1/2) * A * D^(-1/2)
    A_normalized = D_inv_sqrt @ A @ D_inv_sqrt
    
    return A_normalized

# Process landmarks from string to numpy array and create adjacency matrix
def process_landmarks_and_adjacency(landmarks_str):
    """Process landmarks string and create adjacency matrix using distance-based thresholding."""
    try:
        # Parse the landmarks string
        landmarks = np.array(ast.literal_eval(landmarks_str))
        
        # Check if landmarks has the right shape
        if landmarks.ndim != 2 or landmarks.shape[1] != 2:
            print(f"Warning: Unexpected landmark shape: {landmarks.shape}. Reshaping.")
            # Try to fix common shape issues
            if landmarks.size % 2 == 0:  # If we have an even number of values
                landmarks = landmarks.reshape(-1, 2)
            else:
                # Drop the last element if odd
                landmarks = landmarks[:-1].reshape(-1, 2)
        
        # Create adjacency matrix using distance-based thresholding
        # The threshold is determined automatically (20th percentile of distances)
        adjacency_matrix, edges = construct_adjacency_matrix(landmarks, threshold=None)
        
        # Normalize adjacency matrix
        normalized_adjacency = normalize_adjacency_matrix(adjacency_matrix)
        
        return landmarks, normalized_adjacency, edges
        
    except Exception as e:
        print(f"Error processing landmarks: {e}")
        # Return minimal fallback values
        num_landmarks = 68  # Standard number of facial landmarks
        landmarks = np.zeros((num_landmarks, 2))
        adjacency_matrix = np.eye(num_landmarks)
        edges = [(i, (i+1) % num_landmarks) for i in range(num_landmarks)]
        return landmarks, adjacency_matrix, edges

# Custom Dataset class
class LandmarkDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        try:
            row = self.X.iloc[idx]
            
            # Process landmarks and create adjacency matrix
            landmarks, normalized_adjacency, edges = process_landmarks_and_adjacency(row)
            
            # Convert landmarks to tensor (Shape: num_landmarks, 2)
            x = torch.tensor(landmarks, dtype=torch.float)
            num_landmarks = landmarks.shape[0]
            
            # Add additional features: normalized coordinates
            # Scale landmarks to [0,1] range
            x_min, x_max = x[:, 0].min(), x[:, 0].max()
            y_min, y_max = x[:, 1].min(), x[:, 1].max()
            x_range = max(x_max - x_min, 1e-8)  # Avoid division by zero
            y_range = max(y_max - y_min, 1e-8)
            
            # Create normalized coordinates
            x_norm = (x[:, 0] - x_min) / x_range
            y_norm = (x[:, 1] - y_min) / y_range
            
            # Stack with original coordinates to make feature vector
            x = torch.stack([x[:, 0], x[:, 1], x_norm, y_norm], dim=1)
            
            # Convert edges to edge_index format (Shape: 2, num_edges)
            if edges:
                edge_index = torch.tensor([[i, j] for i, j in edges], dtype=torch.long).t()
            else:
                # If no edges were found, create self-loops as a fallback
                edge_index = torch.tensor([[i, i] for i in range(num_landmarks)], dtype=torch.long).t()
            
            # Get the label for this sample
            y = torch.tensor(self.y.iloc[idx], dtype=torch.long)
            
            # Convert adjacency matrix to tensor
            adj = torch.tensor(normalized_adjacency, dtype=torch.float)
            
            return Data(
                x=x, 
                edge_index=edge_index, 
                y=y, 
                adj=adj
            )
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return self._create_fallback_sample()
    
    def _create_fallback_sample(self):
        """Create a fallback sample when processing fails."""
        num_landmarks = 68  # Standard number of facial landmarks
        x = torch.zeros((num_landmarks, 4), dtype=torch.float)  # 4 features
        edge_index = torch.tensor([[i, (i+1) % num_landmarks] for i in range(num_landmarks)], dtype=torch.long).t()
        y = torch.tensor(0, dtype=torch.long)  # Default to first emotion class
        adj = torch.eye(num_landmarks, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y, adj=adj)

# Prepare data
X = df_train['Landmarks']
y = df_train['encoded_label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = X, df_test['Landmarks'], y, df_test['encoded_label']


# Create datasets
train_dataset = LandmarkDataset(X_train, y_train)
test_dataset = LandmarkDataset(X_test, y_test)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Simple GCN model with multiple message-passing layers
class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=8):
        super(SimpleGCN, self).__init__()
        
        self.num_layers = num_layers
        
        # First convolution layer
        self.conv_first = GCNConv(input_dim, hidden_dim)
        
        # Middle message-passing layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Last convolution layer
        self.conv_last = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final prediction layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch, adj=None):
        # First layer
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        x = self.batch_norms[0](x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Middle layers with residual connections
        for i in range(self.num_layers - 2):
            identity = x  # Save for residual connection
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)
            x = self.batch_norms[i+1](x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = x + identity  # Residual connection
        
        # Last layer
        x = self.conv_last(x, edge_index)
        x = F.relu(x)
        
        # Global pooling: combine mean and max pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_global = torch.cat([x_mean, x_max], dim=1)
        
        # Prediction
        x = self.fc1(x_global)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        
        return x

# Create the model
# Determine input dimension from a sample
sample_data = train_dataset[0]
input_dim = sample_data.x.shape[1]  # Number of features per node
hidden_dim = 64  # Hidden dimension size
output_dim = len(emotion_list)  # Number of emotion classes

# Initialize the model
model = SimpleGCN(input_dim, hidden_dim, output_dim, num_layers=8).to(device)

# Print model summary
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Optimizer and Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
criterion = nn.CrossEntropyLoss()

# Training function
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch, batch.adj)
        
        # Compute loss
        loss = criterion(out, batch.y)
        total_loss += loss.item() * batch.num_graphs
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(out, dim=1)
        correct += (predicted == batch.y).sum().item()
        total += batch.y.size(0)
    
    train_loss = total_loss / total
    train_accuracy = correct / total
    
    return train_loss, train_accuracy

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch, batch.adj)
            
            # Compute loss
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            
            # Calculate accuracy
            _, predicted = torch.max(out, dim=1)
            correct += (predicted == batch.y).sum().item()
            total += batch.y.size(0)
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    val_loss = total_loss / total
    val_accuracy = correct / total
    
    return val_loss, val_accuracy, all_preds, all_labels

# Save best model
def save_checkpoint(state, filename="best_custom_gcn.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

# Training loop
epochs = 100
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_acc = 0.0

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Train
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Evaluate
    val_loss, val_accuracy, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    # Update learning rate
    scheduler.step(val_accuracy)
    
    # Save best model
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'accuracy': val_accuracy,
        })
    
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")
    
    # Early stopping
    if epoch >= 20 and all(val_accuracies[-5] >= val_accuracies[-5+i] for i in range(1, 5)):
        print("Early stopping triggered. No improvement in validation accuracy for 5 epochs.")
        break

# Plot training curves
plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('custom_gcn_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Load the best model for final evaluation
checkpoint = torch.load("best_custom_gcn.pth")
model.load_state_dict(checkpoint['state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['accuracy']*100:.2f}%")

# Final evaluation
final_val_loss, final_val_accuracy, final_preds, final_labels = evaluate(model, test_loader, criterion, device)
print(f"Final Test Accuracy: {final_val_accuracy*100:.2f}%")

# Generate confusion matrix
cm = confusion_matrix(final_labels, final_preds)
print("\nConfusion Matrix:")
print(cm)

# Compute classification report
report = classification_report(final_labels, final_preds, target_names=emotion_list)
print("\nClassification Report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_list, yticklabels=emotion_list)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('custom_gcn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize a sample of facial landmarks with predictions
def visualize_landmark_predictions(dataset, model, device, num_samples=6):
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    model.eval()
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        # Get sample
        data = dataset[idx]
        
        # Get landmarks (first two dimensions are x, y coordinates)
        landmarks = data.x[:, :2].cpu().numpy()
        
        # Process one sample at a time properly
        batch = data.clone().to(device)
        batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
        
        # Get prediction
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = torch.argmax(out, dim=1).item()
        
        # Get ground truth
        true_label = data.y.item()
        
        # Plot landmarks
        ax = axes[i]
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', s=30)
        
        # Plot edges
        edge_index = data.edge_index.cpu().numpy()
        for j in range(edge_index.shape[1]):
            src, dst = edge_index[0, j], edge_index[1, j]
            ax.plot([landmarks[src, 0], landmarks[dst, 0]],
                   [landmarks[src, 1], landmarks[dst, 1]], 'gray', alpha=0.5, linewidth=0.5)
        
        # Flip y-axis for correct facial orientation
        ax.invert_yaxis()
        
        # Set title with prediction and ground truth
        correct = pred == true_label
        color = 'green' if correct else 'red'
        ax.set_title(f"True: {emotion_list[true_label]}\nPred: {emotion_list[pred]}", 
                    color=color, fontweight='bold')
        
        # Remove axes for cleaner visualization
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('custom_gcn_landmark_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize predictions
visualize_landmark_predictions(test_dataset, model, device)

# Visualize the graph structure for different thresholds
def visualize_threshold_comparison(dataset, num_thresholds=3, sample_idx=0):
    """
    Visualize how different distance thresholds affect the graph structure.
    
    Args:
        dataset: The dataset containing the samples
        num_thresholds: Number of different thresholds to visualize
        sample_idx: Index of the sample to visualize
    """
    # Get sample data
    data = dataset[sample_idx]
    landmarks = data.x[:, :2].cpu().numpy()
    
    # Define thresholds as percentiles
    percentiles = [10, 20, 40]  # Lower percentile = stricter threshold = fewer connections
    
    fig, axes = plt.subplots(1, len(percentiles), figsize=(15, 5))
    
    # Calculate all pairwise distances
    num_landmarks = len(landmarks)
    distances = np.zeros((num_landmarks, num_landmarks))
    for i in range(num_landmarks):
        for j in range(i+1, num_landmarks):
            dist = np.linalg.norm(landmarks[i] - landmarks[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Get all unique distances (upper triangle)
    unique_distances = distances[np.triu_indices(num_landmarks, k=1)]
    
    for i, p in enumerate(percentiles):
        # Calculate threshold
        threshold = np.percentile(unique_distances, p)
        
        # Create adjacency matrix
        adjacency_matrix = np.zeros((num_landmarks, num_landmarks), dtype=int)
        edges = []
        for i_lm in range(num_landmarks):
            for j_lm in range(i_lm+1, num_landmarks):
                if distances[i_lm, j_lm] < threshold:
                    adjacency_matrix[i_lm, j_lm] = 1
                    adjacency_matrix[j_lm, i_lm] = 1
                    edges.append((i_lm, j_lm))
        
        # Plot
        ax = axes[i]
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', s=30)
        
        # Plot edges
        for src, dst in edges:
            ax.plot([landmarks[src, 0], landmarks[dst, 0]],
                   [landmarks[src, 1], landmarks[dst, 1]], 'gray', alpha=0.5, linewidth=0.5)
        
        # Flip y-axis for correct facial orientation
        ax.invert_yaxis()
        
        # Set title
        ax.set_title(f"Percentile: {p}% (Threshold: {threshold:.2f})\nEdges: {len(edges)}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('threshold_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize different thresholds
visualize_threshold_comparison(test_dataset, num_thresholds=3)

    # Save the final model
torch.save({
    'model_state_dict': model.state_dict(),
    'emotion_list': emotion_list,
    'config': {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'num_layers': 8
    }
}, 'final_distance_gcn_model.pth')

print("Final model saved as 'final_distance_gcn_model.pth'")


# -------------------------------------------------------------------------------------------

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
# from torch.utils.data import Dataset
# from torch_geometric.data import Data, DataLoader
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import ast
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import warnings

# # Suppress specific warning by category
# warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")

# # Set random seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # Load the dataframe
# df = pd.read_csv('tif_df_train.csv')

# # Initialize the label encoder
# emotion_list = ["Angry", "Disgust", "Fear", "Happiness", "Neutral", "Sad", "Surprised"]
# encoder = LabelEncoder()
# encoder.fit(emotion_list)
# df['encoded_label'] = encoder.transform(df['Labels'])

# # Function to construct adjacency matrix using distance-based thresholding
# def construct_adjacency_matrix(landmarks, threshold=None):
#     """
#     Create an adjacency matrix based on Euclidean distances between landmarks.
#     Connections are established when distance < threshold.
    
#     Args:
#         landmarks: Numpy array of shape (num_landmarks, 2) containing landmark coordinates
#         threshold: Distance threshold for connecting landmarks. If None, calculated automatically.
    
#     Returns:
#         adjacency_matrix: Binary matrix where A_ij = 1 if landmarks i and j are connected
#         valid_connections: List of tuples (i,j) representing connected landmarks
#     """
#     num_landmarks = len(landmarks)
    
#     # Create adjacency matrix
#     adjacency_matrix = np.zeros((num_landmarks, num_landmarks), dtype=int)
    
#     # Calculate all pairwise distances between landmarks
#     distances = np.zeros((num_landmarks, num_landmarks))
#     for i in range(num_landmarks):
#         for j in range(i+1, num_landmarks):  # Only calculate upper triangle
#             dist = np.linalg.norm(landmarks[i] - landmarks[j])
#             distances[i, j] = dist
#             distances[j, i] = dist  # Mirror to lower triangle
    
#     # If threshold is not provided, calculate it automatically
#     # Use a percentile of the distance distribution as the threshold
#     if threshold is None:
#         # Flatten the upper triangle (excluding diagonal) to get all unique distances
#         unique_distances = distances[np.triu_indices(num_landmarks, k=1)]
#         # Use a percentile as the threshold (e.g., 20th percentile)
#         threshold = np.percentile(unique_distances, 20)  # Can be adjusted for density control
    
#     # Apply thresholding: create edges between landmarks whose distance is below threshold
#     valid_connections = []
#     for i in range(num_landmarks):
#         for j in range(i+1, num_landmarks):  # Only process upper triangle
#             if distances[i, j] < threshold:
#                 adjacency_matrix[i, j] = 1
#                 adjacency_matrix[j, i] = 1  # Symmetric for undirected graph
#                 valid_connections.append((i, j))
    
#     # Fallback if no valid connections were found
#     if not valid_connections:
#         print("Warning: No connections were made using the threshold. Using fallback chain connections.")
#         for i in range(num_landmarks):
#             next_idx = (i+1) % num_landmarks
#             adjacency_matrix[i, next_idx] = 1
#             adjacency_matrix[next_idx, i] = 1
#             valid_connections.append((i, next_idx))
    
#     return adjacency_matrix, valid_connections

# # Normalize adjacency matrix
# def normalize_adjacency_matrix(A):
#     """
#     Normalize adjacency matrix using symmetric normalization:
#     A_norm = D^(-1/2) * A * D^(-1/2)
#     """
#     # Add self-loops
#     A = A + np.eye(A.shape[0])
    
#     # Calculate degree matrix
#     D = np.sum(A, axis=1)
    
#     # Avoid division by zero
#     D_inv_sqrt = 1.0 / np.sqrt(np.maximum(D, 1e-12))
#     D_inv_sqrt = np.diag(D_inv_sqrt)
    
#     # Normalize using D^(-1/2) * A * D^(-1/2)
#     A_normalized = D_inv_sqrt @ A @ D_inv_sqrt
    
#     return A_normalized

# # Process landmarks from string to numpy array and create adjacency matrix
# def process_landmarks_and_adjacency(landmarks_str):
#     """Process landmarks string and create adjacency matrix using distance-based thresholding."""
#     try:
#         # Parse the landmarks string
#         landmarks = np.array(ast.literal_eval(landmarks_str))
        
#         # Check if landmarks has the right shape
#         if landmarks.ndim != 2 or landmarks.shape[1] != 2:
#             print(f"Warning: Unexpected landmark shape: {landmarks.shape}. Reshaping.")
#             # Try to fix common shape issues
#             if landmarks.size % 2 == 0:  # If we have an even number of values
#                 landmarks = landmarks.reshape(-1, 2)
#             else:
#                 # Drop the last element if odd
#                 landmarks = landmarks[:-1].reshape(-1, 2)
        
#         # Create adjacency matrix using distance-based thresholding
#         # The threshold is determined automatically (20th percentile of distances)
#         adjacency_matrix, edges = construct_adjacency_matrix(landmarks, threshold=None)
        
#         # Normalize adjacency matrix
#         normalized_adjacency = normalize_adjacency_matrix(adjacency_matrix)
        
#         return landmarks, normalized_adjacency, edges
        
#     except Exception as e:
#         print(f"Error processing landmarks: {e}")
#         # Return minimal fallback values
#         num_landmarks = 68  # Standard number of facial landmarks
#         landmarks = np.zeros((num_landmarks, 2))
#         adjacency_matrix = np.eye(num_landmarks)
#         edges = [(i, (i+1) % num_landmarks) for i in range(num_landmarks)]
#         return landmarks, adjacency_matrix, edges

# # Custom Dataset class
# class LandmarkDataset(torch.utils.data.Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         try:
#             row = self.X.iloc[idx]
            
#             # Process landmarks and create adjacency matrix
#             landmarks, normalized_adjacency, edges = process_landmarks_and_adjacency(row)
            
#             # Convert landmarks to tensor (Shape: num_landmarks, 2)
#             x = torch.tensor(landmarks, dtype=torch.float)
#             num_landmarks = landmarks.shape[0]
            
#             # Add additional features: normalized coordinates
#             # Scale landmarks to [0,1] range
#             x_min, x_max = x[:, 0].min(), x[:, 0].max()
#             y_min, y_max = x[:, 1].min(), x[:, 1].max()
#             x_range = max(x_max - x_min, 1e-8)  # Avoid division by zero
#             y_range = max(y_max - y_min, 1e-8)
            
#             # Create normalized coordinates
#             x_norm = (x[:, 0] - x_min) / x_range
#             y_norm = (x[:, 1] - y_min) / y_range
            
#             # Stack with original coordinates to make feature vector
#             x = torch.stack([x[:, 0], x[:, 1], x_norm, y_norm], dim=1)
            
#             # Convert edges to edge_index format (Shape: 2, num_edges)
#             if edges:
#                 edge_index = torch.tensor([[i, j] for i, j in edges], dtype=torch.long).t()
#             else:
#                 # If no edges were found, create self-loops as a fallback
#                 edge_index = torch.tensor([[i, i] for i in range(num_landmarks)], dtype=torch.long).t()
            
#             # Get the label for this sample
#             y = torch.tensor(self.y.iloc[idx], dtype=torch.long)
            
#             # Convert adjacency matrix to tensor
#             adj = torch.tensor(normalized_adjacency, dtype=torch.float)
            
#             return Data(
#                 x=x, 
#                 edge_index=edge_index, 
#                 y=y, 
#                 adj=adj
#             )
            
#         except Exception as e:
#             print(f"Error processing sample {idx}: {e}")
#             return self._create_fallback_sample()
    
#     def _create_fallback_sample(self):
#         """Create a fallback sample when processing fails."""
#         num_landmarks = 68  # Standard number of facial landmarks
#         x = torch.zeros((num_landmarks, 4), dtype=torch.float)  # 4 features
#         edge_index = torch.tensor([[i, (i+1) % num_landmarks] for i in range(num_landmarks)], dtype=torch.long).t()
#         y = torch.tensor(0, dtype=torch.long)  # Default to first emotion class
#         adj = torch.eye(num_landmarks, dtype=torch.float)
#         return Data(x=x, edge_index=edge_index, y=y, adj=adj)

# # Prepare data
# X = df['Landmarks']
# y = df['encoded_label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create datasets
# train_dataset = LandmarkDataset(X_train, y_train)
# test_dataset = LandmarkDataset(X_test, y_test)

# # Create DataLoaders
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # Simple GCN model with multiple message-passing layers
# class SimpleGCN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=8):
#         super(SimpleGCN, self).__init__()
        
#         self.num_layers = num_layers
        
#         # First convolution layer
#         self.conv_first = GCNConv(input_dim, hidden_dim)
        
#         # Middle message-passing layers
#         self.conv_layers = nn.ModuleList()
#         for i in range(num_layers - 2):
#             self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
#         # Last convolution layer
#         self.conv_last = GCNConv(hidden_dim, hidden_dim)
        
#         # Batch normalization layers
#         self.batch_norms = nn.ModuleList()
#         for i in range(num_layers - 1):
#             self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
#         # Final prediction layers
#         self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
        
#     def forward(self, x, edge_index, batch, adj=None):
#         # First layer
#         x = self.conv_first(x, edge_index)
#         x = F.relu(x)
#         x = self.batch_norms[0](x)
#         x = F.dropout(x, p=0.2, training=self.training)
        
#         # Middle layers with residual connections
#         for i in range(self.num_layers - 2):
#             identity = x  # Save for residual connection
#             x = self.conv_layers[i](x, edge_index)
#             x = F.relu(x)
#             x = self.batch_norms[i+1](x)
#             x = F.dropout(x, p=0.2, training=self.training)
#             x = x + identity  # Residual connection
        
#         # Last layer
#         x = self.conv_last(x, edge_index)
#         x = F.relu(x)
        
#         # Global pooling: combine mean and max pooling
#         x_mean = global_mean_pool(x, batch)
#         x_max = global_max_pool(x, batch)
#         x_global = torch.cat([x_mean, x_max], dim=1)
        
#         # Prediction
#         x = self.fc1(x_global)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.3, training=self.training)
#         x = self.fc2(x)
        
#         return x

# # Create the model
# # Determine input dimension from a sample
# sample_data = train_dataset[0]
# input_dim = sample_data.x.shape[1]  # Number of features per node
# hidden_dim = 64  # Hidden dimension size
# output_dim = len(emotion_list)  # Number of emotion classes

# # Initialize the model
# model = SimpleGCN(input_dim, hidden_dim, output_dim, num_layers=8).to(device)

# # Print model summary
# print(model)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total parameters: {total_params}")

# # Define Focal Loss class
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction='mean'):
#         """
#         Focal Loss implementation for multi-class classification.
        
#         Args:
#             alpha (float): Weighting factor, default is 1
#             gamma (float): Focusing parameter, default is 2
#             reduction (str): 'mean', 'sum' or 'none', default is 'mean'
#         """
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
#     def forward(self, inputs, targets):
#         """
#         Forward pass.
        
#         Args:
#             inputs: Model predictions (logits), shape [B, C] where B is batch size and C is number of classes
#             targets: Ground truth labels, shape [B]
            
#         Returns:
#             loss: Computed focal loss
#         """
#         # Get standard cross entropy loss
#         ce_loss = self.cross_entropy(inputs, targets)
        
#         # Get probabilities
#         probs = F.softmax(inputs, dim=1)
#         # Get probability for the correct class
#         p_t = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
        
#         # Apply weighting factor and focusing parameter
#         focal_weights = (1 - p_t) ** self.gamma
#         focal_loss = self.alpha * focal_weights * ce_loss
        
#         # Apply reduction
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

# # Optimizer and Loss function
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# # Initialize Focal Loss instead of Cross Entropy Loss
# # You can adjust alpha and gamma parameters 
# criterion = FocalLoss(alpha=1.0, gamma=2.0)


# # Training function
# def train(model, loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
    
#     for batch in loader:
#         batch = batch.to(device)
        
#         # Zero the gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         out = model(batch.x, batch.edge_index, batch.batch, batch.adj)
        
#         # Compute loss
#         loss = criterion(out, batch.y)
#         total_loss += loss.item() * batch.num_graphs
        
#         # Backward pass and optimize
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         # Calculate accuracy
#         _, predicted = torch.max(out, dim=1)
#         correct += (predicted == batch.y).sum().item()
#         total += batch.y.size(0)
    
#     train_loss = total_loss / total
#     train_accuracy = correct / total
    
#     return train_loss, train_accuracy

# # Evaluation function
# def evaluate(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
            
#             # Forward pass
#             out = model(batch.x, batch.edge_index, batch.batch, batch.adj)
            
#             # Compute loss
#             loss = criterion(out, batch.y)
#             total_loss += loss.item() * batch.num_graphs
            
#             # Calculate accuracy
#             _, predicted = torch.max(out, dim=1)
#             correct += (predicted == batch.y).sum().item()
#             total += batch.y.size(0)
            
#             # Store predictions and labels
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(batch.y.cpu().numpy())
    
#     val_loss = total_loss / total
#     val_accuracy = correct / total
    
#     return val_loss, val_accuracy, all_preds, all_labels

# # Save best model
# def save_checkpoint(state, filename="best_custom_gcn.pth"):
#     torch.save(state, filename)
#     print(f"Checkpoint saved to {filename}")

# # Training loop
# epochs = 100
# train_losses = []
# train_accuracies = []
# val_losses = []
# val_accuracies = []
# best_val_acc = 0.0

# for epoch in range(epochs):
#     print(f"\nEpoch {epoch + 1}/{epochs}")
    
#     # Train
#     train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
#     train_losses.append(train_loss)
#     train_accuracies.append(train_accuracy)
    
#     # Evaluate
#     val_loss, val_accuracy, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
#     val_losses.append(val_loss)
#     val_accuracies.append(val_accuracy)
    
#     # Update learning rate
#     scheduler.step(val_accuracy)
    
#     # Save best model
#     if val_accuracy > best_val_acc:
#         best_val_acc = val_accuracy
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'accuracy': val_accuracy,
#         })
    
#     print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%")
#     print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")
    
#     # Early stopping
#     if epoch >= 20 and all(val_accuracies[-5] >= val_accuracies[-5+i] for i in range(1, 5)):
#         print("Early stopping triggered. No improvement in validation accuracy for 5 epochs.")
#         break

# # Plot training curves
# plt.figure(figsize=(12, 5))

# # Loss curve
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()

# # Accuracy curve
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label='Train Accuracy')
# plt.plot(val_accuracies, label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.savefig('custom_gcn_training_curves.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Load the best model for final evaluation
# checkpoint = torch.load("best_custom_gcn.pth")
# model.load_state_dict(checkpoint['state_dict'])
# print(f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['accuracy']*100:.2f}%")

# # Final evaluation
# final_val_loss, final_val_accuracy, final_preds, final_labels = evaluate(model, test_loader, criterion, device)
# print(f"Final Test Accuracy: {final_val_accuracy*100:.2f}%")

# # Generate confusion matrix
# cm = confusion_matrix(final_labels, final_preds)
# print("\nConfusion Matrix:")
# print(cm)

# # Compute classification report
# report = classification_report(final_labels, final_preds, target_names=emotion_list)
# print("\nClassification Report:")
# print(report)

# # Plot confusion matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_list, yticklabels=emotion_list)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.savefig('custom_gcn_confusion_matrix.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Visualize a sample of facial landmarks with predictions
# def visualize_landmark_predictions(dataset, model, device, num_samples=6):
#     # Get random samples
#     indices = np.random.choice(len(dataset), num_samples, replace=False)
    
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.flatten()
    
#     model.eval()
    
#     for i, idx in enumerate(indices):
#         if i >= len(axes):
#             break
            
#         # Get sample
#         data = dataset[idx]
        
#         # Get landmarks (first two dimensions are x, y coordinates)
#         landmarks = data.x[:, :2].cpu().numpy()
        
#         # Process one sample at a time properly
#         batch = data.clone().to(device)
#         batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
        
#         # Get prediction
#         with torch.no_grad():
#             out = model(batch.x, batch.edge_index, batch.batch)
#             pred = torch.argmax(out, dim=1).item()
        
#         # Get ground truth
#         true_label = data.y.item()
        
#         # Plot landmarks
#         ax = axes[i]
#         ax.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', s=30)
        
#         # Plot edges
#         edge_index = data.edge_index.cpu().numpy()
#         for j in range(edge_index.shape[1]):
#             src, dst = edge_index[0, j], edge_index[1, j]
#             ax.plot([landmarks[src, 0], landmarks[dst, 0]],
#                    [landmarks[src, 1], landmarks[dst, 1]], 'gray', alpha=0.5, linewidth=0.5)
        
#         # Flip y-axis for correct facial orientation
#         ax.invert_yaxis()
        
#         # Set title with prediction and ground truth
#         correct = pred == true_label
#         color = 'green' if correct else 'red'
#         ax.set_title(f"True: {emotion_list[true_label]}\nPred: {emotion_list[pred]}", 
#                     color=color, fontweight='bold')
        
#         # Remove axes for cleaner visualization
#         ax.set_xticks([])
#         ax.set_yticks([])
    
#     plt.tight_layout()
#     plt.savefig('custom_gcn_landmark_predictions.png', dpi=300, bbox_inches='tight')
#     plt.show()

# # Visualize predictions
# visualize_landmark_predictions(test_dataset, model, device)

# # Visualize the graph structure for different thresholds
# def visualize_threshold_comparison(dataset, num_thresholds=3, sample_idx=0):
#     """
#     Visualize how different distance thresholds affect the graph structure.
    
#     Args:
#         dataset: The dataset containing the samples
#         num_thresholds: Number of different thresholds to visualize
#         sample_idx: Index of the sample to visualize
#     """
#     # Get sample data
#     data = dataset[sample_idx]
#     landmarks = data.x[:, :2].cpu().numpy()
    
#     # Define thresholds as percentiles
#     percentiles = [10, 20, 40]  # Lower percentile = stricter threshold = fewer connections
    
#     fig, axes = plt.subplots(1, len(percentiles), figsize=(15, 5))
    
#     # Calculate all pairwise distances
#     num_landmarks = len(landmarks)
#     distances = np.zeros((num_landmarks, num_landmarks))
#     for i in range(num_landmarks):
#         for j in range(i+1, num_landmarks):
#             dist = np.linalg.norm(landmarks[i] - landmarks[j])
#             distances[i, j] = dist
#             distances[j, i] = dist
    
#     # Get all unique distances (upper triangle)
#     unique_distances = distances[np.triu_indices(num_landmarks, k=1)]
    
#     for i, p in enumerate(percentiles):
#         # Calculate threshold
#         threshold = np.percentile(unique_distances, p)
        
#         # Create adjacency matrix
#         adjacency_matrix = np.zeros((num_landmarks, num_landmarks), dtype=int)
#         edges = []
#         for i_lm in range(num_landmarks):
#             for j_lm in range(i_lm+1, num_landmarks):
#                 if distances[i_lm, j_lm] < threshold:
#                     adjacency_matrix[i_lm, j_lm] = 1
#                     adjacency_matrix[j_lm, i_lm] = 1
#                     edges.append((i_lm, j_lm))
        
#         # Plot
#         ax = axes[i]
#         ax.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', s=30)
        
#         # Plot edges
#         for src, dst in edges:
#             ax.plot([landmarks[src, 0], landmarks[dst, 0]],
#                    [landmarks[src, 1], landmarks[dst, 1]], 'gray', alpha=0.5, linewidth=0.5)
        
#         # Flip y-axis for correct facial orientation
#         ax.invert_yaxis()
        
#         # Set title
#         ax.set_title(f"Percentile: {p}% (Threshold: {threshold:.2f})\nEdges: {len(edges)}")
#         ax.set_xticks([])
#         ax.set_yticks([])
    
#     plt.tight_layout()
#     plt.savefig('threshold_comparison.png', dpi=300, bbox_inches='tight')
#     plt.show()

# # Visualize different thresholds
# visualize_threshold_comparison(test_dataset, num_thresholds=3)

#     # Save the final model
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'emotion_list': emotion_list,
#     'config': {
#         'input_dim': input_dim,
#         'hidden_dim': hidden_dim,
#         'output_dim': output_dim,
#         'num_layers': 8
#     }
# }, 'final_distance_gcn_model.pth')

# print("Final model saved as 'final_distance_gcn_model.pth'")

# ---------------------------------------------

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
# from torch.utils.data import Dataset
# from torch_geometric.data import Data, DataLoader
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import ast
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import warnings

# # Suppress specific warning by category
# warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")

# # Set random seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # Load the dataframe
# df = pd.read_csv('tif_df_train.csv')

# # Initialize the label encoder
# emotion_list = ["Angry", "Disgust", "Fear", "Happiness", "Neutral", "Sad", "Surprised"]
# encoder = LabelEncoder()
# encoder.fit(emotion_list)
# df['encoded_label'] = encoder.transform(df['Labels'])

# # Function to construct adjacency matrix using distance-based thresholding
# def construct_adjacency_matrix(landmarks, threshold=None):
#     """
#     Create an adjacency matrix based on Euclidean distances between landmarks.
#     Connections are established when distance < threshold.
    
#     Args:
#         landmarks: Numpy array of shape (num_landmarks, 2) containing landmark coordinates
#         threshold: Distance threshold for connecting landmarks. If None, calculated automatically.
    
#     Returns:
#         adjacency_matrix: Binary matrix where A_ij = 1 if landmarks i and j are connected
#         valid_connections: List of tuples (i,j) representing connected landmarks
#     """
#     num_landmarks = len(landmarks)
    
#     # Create adjacency matrix
#     adjacency_matrix = np.zeros((num_landmarks, num_landmarks), dtype=int)
    
#     # Calculate all pairwise distances between landmarks
#     distances = np.zeros((num_landmarks, num_landmarks))
#     for i in range(num_landmarks):
#         for j in range(i+1, num_landmarks):  # Only calculate upper triangle
#             dist = np.linalg.norm(landmarks[i] - landmarks[j])
#             distances[i, j] = dist
#             distances[j, i] = dist  # Mirror to lower triangle
    
#     # If threshold is not provided, calculate it automatically
#     # Use a percentile of the distance distribution as the threshold
#     if threshold is None:
#         # Flatten the upper triangle (excluding diagonal) to get all unique distances
#         unique_distances = distances[np.triu_indices(num_landmarks, k=1)]
#         # Use a percentile as the threshold (e.g., 20th percentile)
#         threshold = np.percentile(unique_distances, 20)  # Can be adjusted for density control
    
#     # Apply thresholding: create edges between landmarks whose distance is below threshold
#     valid_connections = []
#     for i in range(num_landmarks):
#         for j in range(i+1, num_landmarks):  # Only process upper triangle
#             if distances[i, j] < threshold:
#                 adjacency_matrix[i, j] = 1
#                 adjacency_matrix[j, i] = 1  # Symmetric for undirected graph
#                 valid_connections.append((i, j))
    
#     # Fallback if no valid connections were found
#     if not valid_connections:
#         print("Warning: No connections were made using the threshold. Using fallback chain connections.")
#         for i in range(num_landmarks):
#             next_idx = (i+1) % num_landmarks
#             adjacency_matrix[i, next_idx] = 1
#             adjacency_matrix[next_idx, i] = 1
#             valid_connections.append((i, next_idx))
    
#     return adjacency_matrix, valid_connections

# # Normalize adjacency matrix
# def normalize_adjacency_matrix(A):
#     """
#     Normalize adjacency matrix using symmetric normalization:
#     A_norm = D^(-1/2) * A * D^(-1/2)
#     """
#     # Add self-loops
#     A = A + np.eye(A.shape[0])
    
#     # Calculate degree matrix
#     D = np.sum(A, axis=1)
    
#     # Avoid division by zero
#     D_inv_sqrt = 1.0 / np.sqrt(np.maximum(D, 1e-12))
#     D_inv_sqrt = np.diag(D_inv_sqrt)
    
#     # Normalize using D^(-1/2) * A * D^(-1/2)
#     A_normalized = D_inv_sqrt @ A @ D_inv_sqrt
    
#     return A_normalized

# # Process landmarks from string to numpy array and create adjacency matrix
# def process_landmarks_and_adjacency(landmarks_str):
#     """Process landmarks string and create adjacency matrix using distance-based thresholding."""
#     try:
#         # Parse the landmarks string
#         landmarks = np.array(ast.literal_eval(landmarks_str))
        
#         # Check if landmarks has the right shape
#         if landmarks.ndim != 2 or landmarks.shape[1] != 2:
#             print(f"Warning: Unexpected landmark shape: {landmarks.shape}. Reshaping.")
#             # Try to fix common shape issues
#             if landmarks.size % 2 == 0:  # If we have an even number of values
#                 landmarks = landmarks.reshape(-1, 2)
#             else:
#                 # Drop the last element if odd
#                 landmarks = landmarks[:-1].reshape(-1, 2)
        
#         # Create adjacency matrix using distance-based thresholding
#         # The threshold is determined automatically (20th percentile of distances)
#         adjacency_matrix, edges = construct_adjacency_matrix(landmarks, threshold=None)
        
#         # Normalize adjacency matrix
#         normalized_adjacency = normalize_adjacency_matrix(adjacency_matrix)
        
#         return landmarks, normalized_adjacency, edges
        
#     except Exception as e:
#         print(f"Error processing landmarks: {e}")
#         # Return minimal fallback values
#         num_landmarks = 68  # Standard number of facial landmarks
#         landmarks = np.zeros((num_landmarks, 2))
#         adjacency_matrix = np.eye(num_landmarks)
#         edges = [(i, (i+1) % num_landmarks) for i in range(num_landmarks)]
#         return landmarks, adjacency_matrix, edges

# # Custom Dataset class
# class LandmarkDataset(torch.utils.data.Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         try:
#             row = self.X.iloc[idx]
            
#             # Process landmarks and create adjacency matrix
#             landmarks, normalized_adjacency, edges = process_landmarks_and_adjacency(row)
            
#             # Convert landmarks to tensor (Shape: num_landmarks, 2)
#             x = torch.tensor(landmarks, dtype=torch.float)
#             num_landmarks = landmarks.shape[0]
            
#             # Add additional features: normalized coordinates
#             # Scale landmarks to [0,1] range
#             x_min, x_max = x[:, 0].min(), x[:, 0].max()
#             y_min, y_max = x[:, 1].min(), x[:, 1].max()
#             x_range = max(x_max - x_min, 1e-8)  # Avoid division by zero
#             y_range = max(y_max - y_min, 1e-8)
            
#             # Create normalized coordinates
#             x_norm = (x[:, 0] - x_min) / x_range
#             y_norm = (x[:, 1] - y_min) / y_range
            
#             # Stack with original coordinates to make feature vector
#             x = torch.stack([x[:, 0], x[:, 1], x_norm, y_norm], dim=1)
            
#             # Convert edges to edge_index format (Shape: 2, num_edges)
#             if edges:
#                 edge_index = torch.tensor([[i, j] for i, j in edges], dtype=torch.long).t()
#             else:
#                 # If no edges were found, create self-loops as a fallback
#                 edge_index = torch.tensor([[i, i] for i in range(num_landmarks)], dtype=torch.long).t()
            
#             # Get the label for this sample
#             y = torch.tensor(self.y.iloc[idx], dtype=torch.long)
            
#             # Convert adjacency matrix to tensor
#             adj = torch.tensor(normalized_adjacency, dtype=torch.float)
            
#             return Data(
#                 x=x, 
#                 edge_index=edge_index, 
#                 y=y, 
#                 adj=adj
#             )
            
#         except Exception as e:
#             print(f"Error processing sample {idx}: {e}")
#             return self._create_fallback_sample()
    
#     def _create_fallback_sample(self):
#         """Create a fallback sample when processing fails."""
#         num_landmarks = 68  # Standard number of facial landmarks
#         x = torch.zeros((num_landmarks, 4), dtype=torch.float)  # 4 features
#         edge_index = torch.tensor([[i, (i+1) % num_landmarks] for i in range(num_landmarks)], dtype=torch.long).t()
#         y = torch.tensor(0, dtype=torch.long)  # Default to first emotion class
#         adj = torch.eye(num_landmarks, dtype=torch.float)
#         return Data(x=x, edge_index=edge_index, y=y, adj=adj)

# # Prepare data
# X = df['Landmarks']
# y = df['encoded_label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create datasets
# train_dataset = LandmarkDataset(X_train, y_train)
# test_dataset = LandmarkDataset(X_test, y_test)

# # Create DataLoaders
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# from torch_geometric.nn.pool import global_add_pool
# # Advanced GNN model with Graph Attention and gating mechanisms
# class AdvancedGNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, num_heads=4, dropout=0.2):
#         super(AdvancedGNN, self).__init__()
        
#         from torch_geometric.nn import GATConv, GATv2Conv, GENConv, GCNConv
#         from torch_geometric.nn import GraphNorm, global_mean_pool, global_max_pool
#         from torch_geometric.nn.pool import global_add_pool
        
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.dropout_rate = dropout
        
#         # Input transformation layer
#         self.input_transform = nn.Linear(input_dim, hidden_dim)
        
#         # First layer: GAT attention layer with multiple heads
#         self.gat1 = GATv2Conv(
#             hidden_dim, 
#             hidden_dim // num_heads,
#             heads=num_heads,
#             dropout=dropout,
#             concat=True
#         )
        
#         # Middle GNN layers
#         self.gnn_layers = nn.ModuleList()
#         for i in range(num_layers - 2):
#             # GENConv is a more advanced graph convolutional layer with SoftMax aggregation
#             self.gnn_layers.append(GENConv(
#                 hidden_dim,
#                 hidden_dim,
#                 aggr='softmax',
#                 t=1.0,
#                 learn_t=True,
#                 num_layers=2,
#                 norm='layer'
#             ))
        
#         # Last GNN layer
#         if num_layers > 1:
#             self.gnn_last = GATv2Conv(
#                 hidden_dim, 
#                 hidden_dim // num_heads, 
#                 heads=num_heads, 
#                 dropout=dropout,
#                 concat=True
#             )
        
#         # Layer normalization for each node embedding
#         self.layer_norms = nn.ModuleList()
#         for i in range(num_layers):
#             self.layer_norms.append(GraphNorm(hidden_dim))
        
#         # Gating mechanism - allows model to control information flow
#         self.gate = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Sigmoid()
#         )
        
#         # Global pooling with attention
#         self.global_attention = nn.Sequential(
#             nn.Linear(hidden_dim, 1),
#             nn.Tanh()
#         )
        
#         # Prediction layers with skip connection
#         self.node_predictor = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim)
#         )
        
#         # Final prediction head
#         self.final_predictor = nn.Sequential(
#             nn.Linear(hidden_dim * 3, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim)
#         )
        
#     def forward(self, x, edge_index, batch, adj=None):
#         # Initial feature transformation
#         x = self.input_transform(x)
        
#         # First GAT layer
#         identity = x  # Save for residual connection
#         x = self.gat1(x, edge_index)
#         x = F.leaky_relu(x)
#         x = self.layer_norms[0](x, batch)
#         x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
#         # Apply gate mechanism
#         gate_val = self.gate(x)
#         x = x * gate_val + identity * (1 - gate_val)  # Gated residual connection
        
#         # Middle layers with residual connections
#         for i in range(self.num_layers - 2):
#             identity = x  # Save for residual connection
#             x = self.gnn_layers[i](x, edge_index)
#             x = F.leaky_relu(x)
#             x = self.layer_norms[i+1](x, batch)
#             x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
#             # Apply gate mechanism
#             gate_val = self.gate(x)
#             x = x * gate_val + identity * (1 - gate_val)  # Gated residual connection
        
#         # Last layer if there's more than one layer
#         if self.num_layers > 1:
#             identity = x  # Save for residual connection
#             x = self.gnn_last(x, edge_index)
#             x = F.leaky_relu(x)
#             x = self.layer_norms[-1](x, batch)
            
#             # Apply gate mechanism
#             gate_val = self.gate(x)
#             x = x * gate_val + identity * (1 - gate_val)  # Gated residual connection
        
#         # Multiple pooling strategies for better graph-level representation
#         x_mean = global_mean_pool(x, batch)
#         x_max = global_max_pool(x, batch)
#         x_sum = global_add_pool(x, batch)
        
#         # Combine pooled features
#         x_global = torch.cat([x_mean, x_max, x_sum], dim=1)
        
#         # Final prediction
#         out = self.final_predictor(x_global)
        
#         return out


# # Create the model
# # Determine input dimension from a sample
# sample_data = train_dataset[0]
# input_dim = sample_data.x.shape[1]  # Number of features per node
# hidden_dim = 64  # Hidden dimension size
# output_dim = len(emotion_list)  # Number of emotion classes

# # Initialize the advanced GNN model
# model = AdvancedGNN(
#     input_dim=input_dim, 
#     hidden_dim=hidden_dim, 
#     output_dim=output_dim, 
#     num_layers=3,  # Reduced from 8 as GAT layers are more powerful
#     num_heads=4,   # Number of attention heads
#     dropout=0.2    # Dropout rate
# ).to(device)

# # Print model summary
# print(model)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total parameters: {total_params}")

# # Define Focal Loss class
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction='mean'):
#         """
#         Focal Loss implementation for multi-class classification.
        
#         Args:
#             alpha (float): Weighting factor, default is 1
#             gamma (float): Focusing parameter, default is 2
#             reduction (str): 'mean', 'sum' or 'none', default is 'mean'
#         """
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
#     def forward(self, inputs, targets):
#         """
#         Forward pass.
        
#         Args:
#             inputs: Model predictions (logits), shape [B, C] where B is batch size and C is number of classes
#             targets: Ground truth labels, shape [B]
            
#         Returns:
#             loss: Computed focal loss
#         """
#         # Get standard cross entropy loss
#         ce_loss = self.cross_entropy(inputs, targets)
        
#         # Get probabilities
#         probs = F.softmax(inputs, dim=1)
#         # Get probability for the correct class
#         p_t = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
        
#         # Apply weighting factor and focusing parameter
#         focal_weights = (1 - p_t) ** self.gamma
#         focal_loss = self.alpha * focal_weights * ce_loss
        
#         # Apply reduction
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

# # Optimizer and Loss function
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# # Initialize Focal Loss instead of Cross Entropy Loss
# # You can adjust alpha and gamma parameters 
# criterion = FocalLoss(alpha=1.0, gamma=2.0)


# # Training function
# def train(model, loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
    
#     for batch in loader:
#         batch = batch.to(device)
        
#         # Zero the gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         out = model(batch.x, batch.edge_index, batch.batch, batch.adj)
        
#         # Compute loss
#         loss = criterion(out, batch.y)
#         total_loss += loss.item() * batch.num_graphs
        
#         # Backward pass and optimize
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         # Calculate accuracy
#         _, predicted = torch.max(out, dim=1)
#         correct += (predicted == batch.y).sum().item()
#         total += batch.y.size(0)
    
#     train_loss = total_loss / total
#     train_accuracy = correct / total
    
#     return train_loss, train_accuracy

# # Evaluation function
# def evaluate(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
            
#             # Forward pass
#             out = model(batch.x, batch.edge_index, batch.batch, batch.adj)
            
#             # Compute loss
#             loss = criterion(out, batch.y)
#             total_loss += loss.item() * batch.num_graphs
            
#             # Calculate accuracy
#             _, predicted = torch.max(out, dim=1)
#             correct += (predicted == batch.y).sum().item()
#             total += batch.y.size(0)
            
#             # Store predictions and labels
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(batch.y.cpu().numpy())
    
#     val_loss = total_loss / total
#     val_accuracy = correct / total
    
#     return val_loss, val_accuracy, all_preds, all_labels

# # Save best model
# def save_checkpoint(state, filename="best_custom_gcn.pth"):
#     torch.save(state, filename)
#     print(f"Checkpoint saved to {filename}")

# # Training loop
# epochs = 100
# train_losses = []
# train_accuracies = []
# val_losses = []
# val_accuracies = []
# best_val_acc = 0.0

# for epoch in range(epochs):
#     print(f"\nEpoch {epoch + 1}/{epochs}")
    
#     # Train
#     train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
#     train_losses.append(train_loss)
#     train_accuracies.append(train_accuracy)
    
#     # Evaluate
#     val_loss, val_accuracy, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
#     val_losses.append(val_loss)
#     val_accuracies.append(val_accuracy)
    
#     # Update learning rate
#     scheduler.step(val_accuracy)
    
#     # Save best model
#     if val_accuracy > best_val_acc:
#         best_val_acc = val_accuracy
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'accuracy': val_accuracy,
#         })
    
#     print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%")
#     print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")
    
#     # Early stopping
#     if epoch >= 20 and all(val_accuracies[-5] >= val_accuracies[-5+i] for i in range(1, 5)):
#         print("Early stopping triggered. No improvement in validation accuracy for 5 epochs.")
#         break

# # Plot training curves
# plt.figure(figsize=(12, 5))

# # Loss curve
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()

# # Accuracy curve
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label='Train Accuracy')
# plt.plot(val_accuracies, label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.savefig('custom_gcn_training_curves.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Load the best model for final evaluation
# checkpoint = torch.load("best_custom_gcn.pth")
# model.load_state_dict(checkpoint['state_dict'])
# print(f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['accuracy']*100:.2f}%")

# # Final evaluation
# final_val_loss, final_val_accuracy, final_preds, final_labels = evaluate(model, test_loader, criterion, device)
# print(f"Final Test Accuracy: {final_val_accuracy*100:.2f}%")

# # Generate confusion matrix
# cm = confusion_matrix(final_labels, final_preds)
# print("\nConfusion Matrix:")
# print(cm)

# # Compute classification report
# report = classification_report(final_labels, final_preds, target_names=emotion_list)
# print("\nClassification Report:")
# print(report)

# # Plot confusion matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_list, yticklabels=emotion_list)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.savefig('custom_gcn_confusion_matrix.png', dpi=300, bbox_inches='tight')
# plt.show()

# # Visualize a sample of facial landmarks with predictions
# def visualize_landmark_predictions(dataset, model, device, num_samples=6):
#     # Get random samples
#     indices = np.random.choice(len(dataset), num_samples, replace=False)
    
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.flatten()
    
#     model.eval()
    
#     for i, idx in enumerate(indices):
#         if i >= len(axes):
#             break
            
#         # Get sample
#         data = dataset[idx]
        
#         # Get landmarks (first two dimensions are x, y coordinates)
#         landmarks = data.x[:, :2].cpu().numpy()
        
#         # Process one sample at a time properly
#         batch = data.clone().to(device)
#         batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
        
#         # Get prediction
#         with torch.no_grad():
#             out = model(batch.x, batch.edge_index, batch.batch)
#             pred = torch.argmax(out, dim=1).item()
        
#         # Get ground truth
#         true_label = data.y.item()
        
#         # Plot landmarks
#         ax = axes[i]
#         ax.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', s=30)
        
#         # Plot edges
#         edge_index = data.edge_index.cpu().numpy()
#         for j in range(edge_index.shape[1]):
#             src, dst = edge_index[0, j], edge_index[1, j]
#             ax.plot([landmarks[src, 0], landmarks[dst, 0]],
#                    [landmarks[src, 1], landmarks[dst, 1]], 'gray', alpha=0.5, linewidth=0.5)
        
#         # Flip y-axis for correct facial orientation
#         ax.invert_yaxis()
        
#         # Set title with prediction and ground truth
#         correct = pred == true_label
#         color = 'green' if correct else 'red'
#         ax.set_title(f"True: {emotion_list[true_label]}\nPred: {emotion_list[pred]}", 
#                     color=color, fontweight='bold')
        
#         # Remove axes for cleaner visualization
#         ax.set_xticks([])
#         ax.set_yticks([])
    
#     plt.tight_layout()
#     plt.savefig('custom_gcn_landmark_predictions.png', dpi=300, bbox_inches='tight')
#     plt.show()

# # Visualize predictions
# visualize_landmark_predictions(test_dataset, model, device)

# # Visualize the graph structure for different thresholds
# def visualize_threshold_comparison(dataset, num_thresholds=3, sample_idx=0):
#     """
#     Visualize how different distance thresholds affect the graph structure.
    
#     Args:
#         dataset: The dataset containing the samples
#         num_thresholds: Number of different thresholds to visualize
#         sample_idx: Index of the sample to visualize
#     """
#     # Get sample data
#     data = dataset[sample_idx]
#     landmarks = data.x[:, :2].cpu().numpy()
    
#     # Define thresholds as percentiles
#     percentiles = [10, 20, 40]  # Lower percentile = stricter threshold = fewer connections
    
#     fig, axes = plt.subplots(1, len(percentiles), figsize=(15, 5))
    
#     # Calculate all pairwise distances
#     num_landmarks = len(landmarks)
#     distances = np.zeros((num_landmarks, num_landmarks))
#     for i in range(num_landmarks):
#         for j in range(i+1, num_landmarks):
#             dist = np.linalg.norm(landmarks[i] - landmarks[j])
#             distances[i, j] = dist
#             distances[j, i] = dist
    
#     # Get all unique distances (upper triangle)
#     unique_distances = distances[np.triu_indices(num_landmarks, k=1)]
    
#     for i, p in enumerate(percentiles):
#         # Calculate threshold
#         threshold = np.percentile(unique_distances, p)
        
#         # Create adjacency matrix
#         adjacency_matrix = np.zeros((num_landmarks, num_landmarks), dtype=int)
#         edges = []
#         for i_lm in range(num_landmarks):
#             for j_lm in range(i_lm+1, num_landmarks):
#                 if distances[i_lm, j_lm] < threshold:
#                     adjacency_matrix[i_lm, j_lm] = 1
#                     adjacency_matrix[j_lm, i_lm] = 1
#                     edges.append((i_lm, j_lm))
        
#         # Plot
#         ax = axes[i]
#         ax.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', s=30)
        
#         # Plot edges
#         for src, dst in edges:
#             ax.plot([landmarks[src, 0], landmarks[dst, 0]],
#                    [landmarks[src, 1], landmarks[dst, 1]], 'gray', alpha=0.5, linewidth=0.5)
        
#         # Flip y-axis for correct facial orientation
#         ax.invert_yaxis()
        
#         # Set title
#         ax.set_title(f"Percentile: {p}% (Threshold: {threshold:.2f})\nEdges: {len(edges)}")
#         ax.set_xticks([])
#         ax.set_yticks([])
    
#     plt.tight_layout()
#     plt.savefig('threshold_comparison.png', dpi=300, bbox_inches='tight')
#     plt.show()

# # Visualize different thresholds
# visualize_threshold_comparison(test_dataset, num_thresholds=3)

# # Save the final model
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'emotion_list': emotion_list,
#     'config': {
#         'input_dim': input_dim,
#         'hidden_dim': hidden_dim,
#         'output_dim': output_dim,
#         'num_layers': 3,
#         'num_heads': 4,
#         'dropout': 0.2
#     }
# }, 'final_advanced_gnn_model.pth')

# print("Final model saved as 'final_advanced_gnn_model.pth'")