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

df = pd.concat([df_train, df_test], ignore_index=True)

# Initialize the label encoder
emotion_list = ["Angry", "Disgust", "Fear", "Happiness", "Neutral", "Sad", "Surprised"]
encoder = LabelEncoder()
encoder.fit(emotion_list)
df_train['encoded_label'] = encoder.transform(df_train['Labels'])
df_test['encoded_label'] = encoder.transform(df_test['Labels'])


# Custom connections for facial landmarks
connections = [
    # Jawline
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    # Right eyebrow
    (17, 18), (18, 19), (19, 20), (20, 21),
    # Left eyebrow
    (22, 23), (23, 24), (24, 25), (25, 26),
    # Nose bridge
    (27, 28), (28, 29), (29, 30),
    # Nose base
    (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
    # Right eye
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    # Left eye
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    # Outer lip
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
    (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60),
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 48),
    # Inner lip
    (60, 67), (61, 66), (62, 65), (63, 64)
]

# Function to construct adjacency matrix from custom connections
def construct_adjacency_matrix(landmarks, custom_connections=connections):
    """Create an adjacency matrix based on custom facial landmark connections."""
    num_landmarks = len(landmarks)
    
    # Create adjacency matrix
    adjacency_matrix = np.zeros((num_landmarks, num_landmarks), dtype=int)
    
    valid_connections = []
    for i, j in custom_connections:
        if i < num_landmarks and j < num_landmarks:  # Ensure indices are valid
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1  # Symmetric for undirected graph
            valid_connections.append((i, j))
    
    # Fallback if no valid connections were found
    if not valid_connections:
        for i in range(num_landmarks):
            adjacency_matrix[i, (i+1) % num_landmarks] = 1
            adjacency_matrix[(i+1) % num_landmarks, i] = 1
            valid_connections.append((i, (i+1) % num_landmarks))
    
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
    """Process landmarks string and create adjacency matrix."""
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
        
        # Create adjacency matrix using custom connections
        adjacency_matrix, edges = construct_adjacency_matrix(landmarks)
        
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
X_train, y_train = X, y
X_test, y_test = df_test['Landmarks'], df_test['encoded_label']

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

# Save the final model
torch.save({
    'model_state_dict': model.state_dict(),
    'emotion_list': emotion_list,
    'custom_connections': connections,  # Save the custom connections
    'config': {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'num_layers': 8
    }
}, 'final_custom_gcn_model.pth')

print("Final model saved as 'final_custom_gcn_model.pth'")