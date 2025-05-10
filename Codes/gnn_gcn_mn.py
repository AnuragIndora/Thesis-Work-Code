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
from scipy.spatial import Delaunay
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

# Function to perform Delaunay triangulation
def delaunay_triangulation(landmarks):
    """Generate edges using Delaunay triangulation on facial landmarks."""
    try:
        tri = Delaunay(landmarks)
        triangles = tri.simplices
        edges = set()
        for triangle in triangles:
            for i in range(3):
                edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
                edges.add(edge)
        return list(edges)
    except Exception as e:
        print(f"Delaunay triangulation failed: {e}")
        return []

# Function to construct adjacency matrix with master node
def construct_master_node_adjacency(landmarks):
    """
    Create an adjacency matrix with a master node that connects to all landmarks.
    The master node serves as a global information aggregator.
    """
    num_landmarks = len(landmarks)
    num_nodes = num_landmarks + 1  # +1 for master node
    
    # Create base adjacency matrix using Delaunay triangulation
    delaunay_edges = delaunay_triangulation(landmarks)
    
    # If triangulation failed, create simple chain connections
    if not delaunay_edges:
        delaunay_edges = [(i, (i+1) % num_landmarks) for i in range(num_landmarks)]
        # Add some cross-connections for better connectivity
        if num_landmarks > 4:
            for i in range(0, num_landmarks, num_landmarks // 4):
                delaunay_edges.append((i, (i + num_landmarks // 2) % num_landmarks))
    
    # Create expanded adjacency matrix with master node
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    # Add regular connections
    for i, j in delaunay_edges:
        if i < num_landmarks and j < num_landmarks:  # Ensure indices are valid
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1  # Symmetric for undirected graph
    
    # Add connections from master node to all other nodes
    master_idx = num_landmarks  # Last index is the master node
    for i in range(num_landmarks):
        adjacency_matrix[master_idx, i] = 1
        adjacency_matrix[i, master_idx] = 1
    
    # Generate edge list from adjacency matrix
    edges = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):  # Upper triangular to avoid duplicates
            if adjacency_matrix[i, j] == 1:
                edges.append((i, j))
                edges.append((j, i))  # Add both directions for undirected graph
    
    return adjacency_matrix, edges

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
    """Process landmarks string and create master node adjacency matrix."""
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
        
        # Create adjacency matrix with master node
        adjacency_matrix, edges = construct_master_node_adjacency(landmarks)
        
        # Normalize adjacency matrix
        normalized_adjacency = normalize_adjacency_matrix(adjacency_matrix)
        
        return landmarks, normalized_adjacency, edges
        
    except Exception as e:
        print(f"Error processing landmarks: {e}")
        # Return minimal fallback values
        num_landmarks = 10  # Arbitrary small number
        landmarks = np.zeros((num_landmarks, 2))
        # Create adjacency matrix with master node
        num_nodes = num_landmarks + 1
        adjacency_matrix = np.eye(num_nodes)
        # Connect master node to all landmarks
        master_idx = num_landmarks
        for i in range(num_landmarks):
            adjacency_matrix[master_idx, i] = 1
            adjacency_matrix[i, master_idx] = 1
        # Create edge list
        edges = [(i, master_idx) for i in range(num_landmarks)] + [(master_idx, i) for i in range(num_landmarks)]
        return landmarks, adjacency_matrix, edges

# Custom Dataset class with master node
class LandmarkDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        try:
            row = self.X.iloc[idx]
            
            # Process landmarks and create adjacency matrix with master node
            landmarks, normalized_adjacency, edges = process_landmarks_and_adjacency(row)
            
            # Convert landmarks to tensor (Shape: num_landmarks, 2)
            x_landmarks = torch.tensor(landmarks, dtype=torch.float)
            num_landmarks = landmarks.shape[0]
            
            # Add additional features: normalized coordinates
            # Scale landmarks to [0,1] range
            x_min, x_max = x_landmarks[:, 0].min(), x_landmarks[:, 0].max()
            y_min, y_max = x_landmarks[:, 1].min(), x_landmarks[:, 1].max()
            x_range = max(x_max - x_min, 1e-8)  # Avoid division by zero
            y_range = max(y_max - y_min, 1e-8)
            
            # Create normalized coordinates
            x_norm = (x_landmarks[:, 0] - x_min) / x_range
            y_norm = (x_landmarks[:, 1] - y_min) / y_range
            
            # Stack with original coordinates to make feature vector for landmarks
            x_landmarks = torch.stack([x_landmarks[:, 0], x_landmarks[:, 1], x_norm, y_norm], dim=1)
            
            # Create feature for master node
            # Use average of all landmarks as position, plus zeros for normalized coords
            master_feature = torch.zeros(4, dtype=torch.float)
            master_feature[0] = x_landmarks[:, 0].mean()  # Average x-coordinate
            master_feature[1] = x_landmarks[:, 1].mean()  # Average y-coordinate
            master_feature[2] = 0.5  # Center of normalized x-range
            master_feature[3] = 0.5  # Center of normalized y-range
            
            # Combine landmark features with master node feature
            x = torch.cat([x_landmarks, master_feature.unsqueeze(0)], dim=0)
            
            # Convert edges to edge_index format (Shape: 2, num_edges)
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t()
            else:
                # If no edges were found, create self-loops as a fallback
                num_nodes = num_landmarks + 1
                edge_index = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t()
            
            # Get the label for this sample
            y = torch.tensor(self.y.iloc[idx], dtype=torch.long)
            
            # Convert adjacency matrix to tensor
            adj = torch.tensor(normalized_adjacency, dtype=torch.float)
            
            # Create a mask to identify the master node (1 for master, 0 for landmarks)
            master_mask = torch.zeros(x.size(0), dtype=torch.bool)
            master_mask[-1] = True  # Last node is the master node
            
            return Data(
                x=x, 
                edge_index=edge_index, 
                y=y, 
                adj=adj,
                master_mask=master_mask
            )
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return self._create_fallback_sample()
    
    def _create_fallback_sample(self):
        """Create a fallback sample when processing fails."""
        num_landmarks = 10  # Small arbitrary number
        num_nodes = num_landmarks + 1  # +1 for master node
        
        # Create features with 4 dimensions
        x = torch.zeros((num_nodes, 4), dtype=torch.float)
        
        # Create edges connecting master node to all landmarks
        master_idx = num_landmarks
        edges = []
        for i in range(num_landmarks):
            edges.append([i, master_idx])
            edges.append([master_idx, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # Default label
        y = torch.tensor(0, dtype=torch.long)
        
        # Create adjacency matrix with master node
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        for i in range(num_landmarks):
            adj[i, master_idx] = 1
            adj[master_idx, i] = 1
        # Add self-loops
        for i in range(num_nodes):
            adj[i, i] = 1
        
        # Normalize adjacency matrix
        D = adj.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(D, min=1e-12)))
        adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt
        
        # Create master node mask
        master_mask = torch.zeros(num_nodes, dtype=torch.bool)
        master_mask[-1] = True
        
        return Data(x=x, edge_index=edge_index, y=y, adj=adj_normalized, master_mask=master_mask)

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

# Check a sample to determine input dimensions
sample_data = train_dataset[0]
input_dim = sample_data.x.shape[1]  # Number of features per node
print(f"Input dimension: {input_dim}")

# Simple GCN model with master node awareness
class MasterNodeGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=8):
        super(MasterNodeGCN, self).__init__()
        
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
        
        # Special attention for master node
        self.master_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Final prediction layers
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch, master_mask=None, adj=None):
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
        
        # Separate master node features using the mask
        if master_mask is None:
            # If mask not provided, assume last node of each graph is master
            master_indices = torch.tensor(
                [offset + num_nodes - 1 for offset, num_nodes in enumerate(batch.bincount())],
                device=x.device
            )
            master_features = x[master_indices]
        else:
            # Get master node features for each graph using the mask
            master_node_idx = torch.nonzero(master_mask, as_tuple=False).view(-1)
            master_features = global_mean_pool(x[master_node_idx], batch[master_node_idx])
        
        # Apply attention to master node features
        master_attention = self.master_attention(master_features)
        master_features = master_features * master_attention
        
        # Global pooling of non-master nodes
        regular_mask = ~master_mask if master_mask is not None else None
        if regular_mask is not None:
            regular_node_idx = torch.nonzero(regular_mask, as_tuple=False).view(-1)
            x_mean = global_mean_pool(x[regular_node_idx], batch[regular_node_idx])
            x_max = global_max_pool(x[regular_node_idx], batch[regular_node_idx])
        else:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
        
        # Concatenate global features with master node features
        x_global = torch.cat([x_mean, x_max, master_features], dim=1)
        
        # Prediction
        x = self.fc1(x_global)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        
        return x

# Initialize model
hidden_dim = 64
output_dim = len(emotion_list)
model = MasterNodeGCN(input_dim, hidden_dim, output_dim, num_layers=8).to(device)

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
        out = model(batch.x, batch.edge_index, batch.batch, batch.master_mask, batch.adj)
        
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
            out = model(batch.x, batch.edge_index, batch.batch, batch.master_mask, batch.adj)
            
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
def save_checkpoint(state, filename="best_master_node_gcn.pth"):
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
plt.savefig('master_node_gcn_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Load the best model for final evaluation
checkpoint = torch.load("best_master_node_gcn.pth")
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
plt.savefig('master_node_gcn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize a sample of facial landmarks with predictions, highlighting the master node
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
        
        # Process one sample at a time properly
        batch = data.clone().to(device)
        batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
        
        # Get prediction
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch, batch.master_mask, batch.adj)
            pred = torch.argmax(out, dim=1).item()
        
        # Get ground truth
        true_label = data.y.item()
        
        # Get landmarks and master node (first two dimensions are x, y coordinates)
        all_nodes = data.x[:, :2].cpu().numpy()
        landmarks = all_nodes[:-1]  # All but the last node
        master_node = all_nodes[-1]  # Last node is the master node
        
        # Plot landmarks
        ax = axes[i]
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', s=30, label='Landmarks')
        
        # Plot master node with a different color and size
        ax.scatter(master_node[0], master_node[1], c='red', s=100, label='Master Node')
        
        # Plot edges
        edge_index = data.edge_index.cpu().numpy()
        for j in range(edge_index.shape[1]):
            src, dst = edge_index[0, j], edge_index[1, j]
            # Check if this edge connects to the master node
            if src == len(landmarks) or dst == len(landmarks):
                color = 'red'
                alpha = 0.3
                linewidth = 0.5
            else:
                color = 'gray'
                alpha = 0.5
                linewidth = 0.5
            ax.plot([all_nodes[src, 0], all_nodes[dst, 0]],
                   [all_nodes[src, 1], all_nodes[dst, 1]], color=color, alpha=alpha, linewidth=linewidth)
        
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
        
        # Add legend to the first plot only
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('master_node_gcn_landmark_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize predictions with master node
visualize_landmark_predictions(test_dataset, model, device)

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
}, 'final_master_node_gcn_model.pth')

print("Final model saved as 'final_master_node_gcn_model.pth'")