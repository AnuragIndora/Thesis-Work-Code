# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
# from torch.utils.data import Dataset
# from torch_geometric.data import DataLoader
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from torch_geometric.data import Data
# from scipy.spatial import Delaunay
# import ast 

# import warnings

# # Suppress specific warning by category (e.g., deprecation warnings)
# warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")

# # Load the dataframe
# df = pd.read_csv('tif_df_train.csv')

# # Initialize the label encoder
# emotion_list = ["Angry", "Disgust", "Fear", "Happiness", "Neutral", "Sad", "Surprised"]
# encoder = LabelEncoder()
# df['encoded_label'] = encoder.fit_transform(df['Labels'])



# # Function to perform Delaunay triangulation
# def delaunay_triangulation(landmarks):
#     tri = Delaunay(landmarks)
#     triangles = tri.simplices
#     edges = set()
#     for triangle in triangles:
#         for i in range(3):
#             edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
#             edges.add(edge)
#     return triangles, list(edges)

# # Function to construct adjacency matrix from Delaunay triangulation
# def construct_adjacency_matrix(landmarks):
#     _, edges = delaunay_triangulation(landmarks)
#     num_landmarks = len(landmarks)
#     adjacency_matrix = np.zeros((num_landmarks, num_landmarks), dtype=int)
    
#     for edge in edges:
#         i, j = edge
#         adjacency_matrix[i, j] = 1
#         adjacency_matrix[j, i] = 1  # Symmetric for undirected graph

#     return adjacency_matrix

# def normalize_adjacency_matrix(A):
#     A = A + np.eye(A.shape[0])  # Add self-connections
#     D = np.diag(np.sum(A, axis=1))
#     D_inv_sqrt = np.linalg.inv(np.sqrt(D))
#     A_normalized = D_inv_sqrt @ A @ D_inv_sqrt
#     return A_normalized


# # # Convert the landmarks into numpy arrays and adjacency matrices
# # def process_landmarks_and_adjacency(row):
# #     landmarks = np.fromstring(row['Landmarks'], sep=' ').reshape(68, 2)  # Shape (68, 2)
# #     adjacency_matrix = construct_adjacency_matrix(landmarks)  # Get adjacency matrix
# #     return landmarks, adjacency_matrix

# # Prepare data
# X = df['Landmarks']
# y = df['encoded_label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Function to process landmarks and adjacency matrix
# def process_landmarks_and_adjacency(landmarks_str):
#     # Parse the string to get the list of landmarks
#     landmarks = np.array(ast.literal_eval(landmarks_str))  # Convert string to list using literal_eval
#     adjacency_matrix = construct_adjacency_matrix(landmarks)  # Get adjacency matrix
#     return landmarks, adjacency_matrix

# # Your existing Dataset class definition remains the same:
# class LandmarkDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         row = self.X.iloc[idx]
        
#         # Process the landmarks and adjacency matrix
#         landmarks, adjacency_matrix = process_landmarks_and_adjacency(row)  # Pass the string directly
        
#         # Convert landmarks to tensor (Shape: 68, 2)
#         x = torch.tensor(landmarks, dtype=torch.float)  
        
#         # Convert adjacency matrix to edge_index (Shape: 2, num_edges)
#         edge_index = torch.tensor(np.array(np.nonzero(adjacency_matrix)), dtype=torch.long) 
        
#         # Get the label for this sample
#         y = torch.tensor(self.y.iloc[idx], dtype=torch.long)  # Label
        
#         return Data(x=x, edge_index=edge_index, y=y)
        
# # Create DataLoader from PyTorch Geometric
# train_dataset = LandmarkDataset(X_train, y_train)
# test_dataset = LandmarkDataset(X_test, y_test)

# # Use the PyTorch Geometric DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Define the GCN-based model
# class GCNModel(nn.Module):
#     def __init__(self, num_nodes=68, num_features=2, num_classes=7):
#         super(GCNModel, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = GCNConv(num_features, 64)
#         self.conv2 = GCNConv(64, 128)
#         self.conv3 = GCNConv(128, 256)
#         self.conv4 = GCNConv(256, 512)
#         self.conv5 = GCNConv(512, 1024)
#         self.conv6 = GCNConv(1024, 2048)
        
#         # Fully connected layers after flattening
#         self.fc1 = nn.Linear(2048, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, num_classes)
        
#     def forward(self, x, edge_index, batch):
#         # Apply GCN layers with ReLU activations
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = F.relu(self.conv3(x, edge_index))
#         x = F.relu(self.conv4(x, edge_index))
#         x = F.relu(self.conv5(x, edge_index))
        
#         # Pooling layers - applying both mean and max pooling
#         x_mean = global_mean_pool(x, batch)  # Global mean pooling
#         x_max = global_max_pool(x, batch)    # Global max pooling
        
#         # Concatenate the pooled results
#         x = torch.cat([x_mean, x_max], dim=1)
        
#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
        
#         return x
        
# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# # Initialize model
# model = GCNModel(num_nodes=68, num_features=2, num_classes=7).to(device)

# # Optimizer and Loss
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# # # Training loop
# # def train(model, loader, optimizer, criterion):
# #     model.train()
# #     total_loss = 0
# #     all_labels = []
# #     all_preds = []
# #     for batch in loader:
# #         batch = batch.to(device)
        
# #         # Zero the gradients
# #         optimizer.zero_grad()
        
# #         # Forward pass
# #         out = model(batch.x, batch.edge_index, batch.batch)
        
# #         # Compute loss
# #         loss = criterion(out, batch.y)
# #         total_loss += loss.item()
        
# #         # Backward pass and optimize
# #         loss.backward()
# #         optimizer.step()
    
# #     return total_loss / len(loader)

# # # Testing loop
# # def test(model, loader):
# #     model.eval()
# #     correct = 0
# #     total = 0
# #     with torch.no_grad():
# #         for batch in loader:
# #             batch = batch.to(device)
# #             out = model(batch.x, batch.edge_index, batch.batch)
# #             _, predicted = torch.max(out, dim=1)
# #             correct += (predicted == batch.y).sum().item()
# #             total += batch.y.size(0)
    
# #     return correct / total

# # # Training and evaluation
# # epochs = 50
# # for epoch in range(epochs):
# #     train_loss = train(model, train_loader, optimizer, criterion)
# #     test_acc = test(model, test_loader)
    
# #     print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%')

# import torch
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import numpy as np

# # Training loop with training accuracy
# def train(model, loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#     all_labels = []
#     all_preds = []
    
#     for batch in loader:
#         batch = batch.to(device)
        
#         # Zero the gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         out = model(batch.x, batch.edge_index, batch.batch)
        
#         # Compute loss
#         loss = criterion(out, batch.y)
#         total_loss += loss.item()
        
#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()
        
#         # Calculate accuracy
#         _, predicted = torch.max(out, dim=1)
#         correct += (predicted == batch.y).sum().item()
#         total += batch.y.size(0)
        
#         # Store predictions and labels for confusion matrix later
#         all_labels.extend(batch.y.cpu().numpy())
#         all_preds.extend(predicted.cpu().numpy())
    
#     train_loss = total_loss / len(loader)
#     train_accuracy = correct / total
#     # Compute confusion matrix
#     cm = confusion_matrix(all_labels, all_preds)
    
#     return train_loss, train_accuracy, cm

# # Testing loop with validation accuracy and loss
# def test(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total = 0
#     all_labels = []
#     all_preds = []
    
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
#             out = model(batch.x, batch.edge_index, batch.batch)
            
#             # Compute loss
#             loss = criterion(out, batch.y)
#             total_loss += loss.item()
            
#             # Calculate accuracy
#             _, predicted = torch.max(out, dim=1)
#             correct += (predicted == batch.y).sum().item()
#             total += batch.y.size(0)
            
#             # Store predictions and labels for confusion matrix later
#             all_labels.extend(batch.y.cpu().numpy())
#             all_preds.extend(predicted.cpu().numpy())
    
#     val_loss = total_loss / len(loader)
#     val_accuracy = correct / total
#     # Compute confusion matrix
#     cm = confusion_matrix(all_labels, all_preds)
    
#     return val_loss, val_accuracy, cm

# # Training and evaluation
# epochs = 50
# train_losses = []
# train_accuracies = []
# val_losses = []
# val_accuracies = []
# cm_list = []  # List to store confusion matrices

# for epoch in range(epochs):
#     print(f"\nEpoch {epoch + 1}/{epochs}")
    
#     # Train
#     train_loss, train_accuracy, cm_train = train(model, train_loader, optimizer, criterion, device)
#     train_losses.append(train_loss)
#     train_accuracies.append(train_accuracy)
    
#     # Validate
#     val_loss, val_accuracy, cm_val = test(model, test_loader, criterion, device)
#     val_losses.append(val_loss)
#     val_accuracies.append(val_accuracy)
#     cm_list.append((cm_train, cm_val))
    
#     print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%")
#     print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")
    
# # Plot loss curve
# plt.figure(figsize=(10, 5))
# plt.plot(range(epochs), train_losses, label='Train Loss')
# plt.plot(range(epochs), val_losses, label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss Curve')
# plt.legend()
# plt.show()

# # Plot accuracy curve
# plt.figure(figsize=(10, 5))
# plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
# plt.plot(range(epochs), val_accuracies, label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Accuracy Curve')
# plt.legend()
# plt.show()

# # Confusion matrix visualization for the final epoch
# cm_train_final, cm_val_final = cm_list[-1]
# fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# sns.heatmap(cm_train_final, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False)
# ax[0].set_title('Training Confusion Matrix')
# ax[0].set_xlabel('Predicted')
# ax[0].set_ylabel('True')
# sns.heatmap(cm_val_final, annot=True, fmt='d', cmap='Blues', ax=ax[1], cbar=False)
# ax[1].set_title('Validation Confusion Matrix')
# ax[1].set_xlabel('Predicted')
# ax[1].set_ylabel('True')
# plt.show()

# ----

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, ChebConv, GraphConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_dense_adj
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
df = pd.read_csv('tif_df_train.csv')

# Initialize the label encoder
emotion_list = ["Angry", "Disgust", "Fear", "Happiness", "Neutral", "Sad", "Surprised"]
encoder = LabelEncoder()
encoder.fit(emotion_list)
df['encoded_label'] = encoder.transform(df['Labels'])

# Define facial landmark regions (adaptive to available landmarks)
def get_facial_regions(num_landmarks):
    """
    Define facial landmark regions adapted to the actual number of landmarks available.
    """
    if num_landmarks == 68:
        # Standard 68-point facial landmarks
        return {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_tip': list(range(31, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'outer_lips': list(range(48, 60)),
            'inner_lips': list(range(60, 68))
        }
    else:
        # For non-standard number of landmarks, create approximate regions
        # Divide landmarks roughly into regions based on proportions
        region_proportions = {
            'jaw': 0.25,           # ~25% of landmarks for jaw
            'eyes': 0.25,          # ~25% for eyes (combines eyes and eyebrows)
            'nose': 0.20,          # ~20% for nose
            'mouth': 0.30          # ~30% for mouth
        }
        
        regions = {}
        start_idx = 0
        
        for region, proportion in region_proportions.items():
            end_idx = start_idx + max(1, int(num_landmarks * proportion))
            end_idx = min(end_idx, num_landmarks)  # Don't exceed the number of landmarks
            regions[region] = list(range(start_idx, end_idx))
            start_idx = end_idx
            
            # If we've reached the end, stop assigning regions
            if start_idx >= num_landmarks:
                break
        
        return regions

# Function to create predefined facial landmark connections
def create_facial_landmark_edges(num_landmarks):
    """
    Create predefined connections between facial landmarks based on anatomical structure,
    adapting to the actual number of landmarks detected.
    Returns a list of tuples representing connected landmarks.
    """
    edges = []
    
    # If we have fewer than expected landmarks, adjust our connections
    if num_landmarks < 68:
        # Create a simple connectivity pattern: connect each landmark to its neighbors
        # This is a fallback when we don't have the standard 68 landmarks
        for i in range(num_landmarks - 1):
            edges.append((i, i+1))
        
        # Add some cross connections for better graph connectivity
        step = max(1, num_landmarks // 10)  # Connect roughly every 10% of points
        for i in range(0, num_landmarks - step, step):
            edges.append((i, i + step))
        
        # Add some long-range connections
        if num_landmarks > 10:
            for i in range(0, num_landmarks // 2):
                edges.append((i, num_landmarks - i - 1))
        
        return edges
    
    # Standard 68-point facial landmark connections
    # Connect jaw points
    for i in range(16):
        edges.append((i, i+1))
    
    # Connect eyebrows
    for i in range(17, 21):
        edges.append((i, i+1))
    for i in range(22, 26):
        edges.append((i, i+1))
    
    # Connect nose bridge
    for i in range(27, 30):
        edges.append((i, i+1))
    
    # Connect nose tip
    for i in range(31, 35):
        edges.append((i, i+1))
    edges.append((35, 31))
    
    # Connect eyes
    for i in range(36, 41):
        edges.append((i, i+1))
    edges.append((41, 36))
    
    for i in range(42, 47):
        edges.append((i, i+1))
    edges.append((47, 42))
    
    # Connect outer lips
    for i in range(48, 59):
        edges.append((i, i+1))
    edges.append((59, 48))
    
    # Connect inner lips
    for i in range(60, 67):
        edges.append((i, i+1))
    edges.append((67, 60))
    
    # Connect between regions (examples)
    # Eyes to eyebrows
    edges.append((19, 37))  # Right eyebrow to right eye
    edges.append((24, 44))  # Left eyebrow to left eye
    
    # Nose to eyes
    edges.append((30, 39))  # Nose bridge to right eye
    edges.append((30, 42))  # Nose bridge to left eye
    
    # Mouth to nose
    edges.append((33, 51))  # Nose tip to top lip
    edges.append((33, 57))  # Nose tip to bottom lip
    
    return edges

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

# Function to construct enhanced adjacency matrix
def construct_enhanced_adjacency_matrix(landmarks):
    """
    Construct an enhanced adjacency matrix using both predefined facial connections
    and Delaunay triangulation to capture spatial relationships.
    Adapts to the actual number of landmarks available.
    """
    num_landmarks = len(landmarks)
    
    # Get edges from facial anatomy (adaptive to number of landmarks)
    facial_edges = create_facial_landmark_edges(num_landmarks)
    
    # Get edges from Delaunay triangulation as a backup/supplement
    try:
        delaunay_edges = delaunay_triangulation(landmarks)
    except Exception as e:
        print(f"Delaunay triangulation failed: {e}. Using only facial edges.")
        delaunay_edges = []
    
    # Combine both sets of edges, ensuring all indices are valid
    all_edges = []
    for i, j in set(facial_edges + delaunay_edges):
        if i < num_landmarks and j < num_landmarks:  # Ensure indices are valid
            all_edges.append((i, j))
    
    # Create adjacency matrix
    adjacency_matrix = np.zeros((num_landmarks, num_landmarks), dtype=int)
    
    for i, j in all_edges:
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  # Symmetric for undirected graph
    
    return adjacency_matrix, all_edges

# Normalize adjacency matrix using symmetric normalization
def normalize_adjacency_matrix(A):
    """
    Normalize adjacency matrix using symmetric normalization method:
    A_norm = D^(-1/2) * A * D^(-1/2)
    """
    # Add self-loops
    A = A + np.eye(A.shape[0])
    
    # Calculate degree matrix
    D = np.sum(A, axis=1)
    
    # Calculate D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    
    # Normalize using D^(-1/2) * A * D^(-1/2)
    A_normalized = D_inv_sqrt @ A @ D_inv_sqrt
    
    return A_normalized

# Function to process landmarks and create adjacency matrix
def process_landmarks_and_adjacency(landmarks_str):
    """Process landmarks string and create enhanced adjacency matrix."""
    try:
        # Parse the string to get the list of landmarks
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
        
        # Get number of landmarks
        num_landmarks = landmarks.shape[0]
        
        # Create enhanced adjacency matrix
        adjacency_matrix, edges = construct_enhanced_adjacency_matrix(landmarks)
        
        # Normalize adjacency matrix
        normalized_adjacency = normalize_adjacency_matrix(adjacency_matrix)
        
        return landmarks, normalized_adjacency, edges
        
    except Exception as e:
        print(f"Error processing landmarks: {e}")
        # Return minimal fallback values
        num_landmarks = 10  # Arbitrary small number
        landmarks = np.zeros((num_landmarks, 2))
        adjacency_matrix = np.eye(num_landmarks)
        edges = [(i, (i+1) % num_landmarks) for i in range(num_landmarks)]
        return landmarks, adjacency_matrix, edges

# Custom Dataset class
class LandmarkDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, add_features=True):
        self.X = X
        self.y = y
        self.add_features = add_features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        
        try:
            # Process the landmarks and adjacency matrix
            landmarks, normalized_adjacency, edges = process_landmarks_and_adjacency(row)
            
            # Convert landmarks to tensor (Shape: num_landmarks, 2)
            x = torch.tensor(landmarks, dtype=torch.float)
            
            num_landmarks = landmarks.shape[0]
            
            if self.add_features:
                # Add additional node features that are adaptive to the number of landmarks
                # This helps capture spatial relationships between landmarks
                
                # Calculate center of mass and relative positions
                center = landmarks.mean(axis=0)
                rel_positions = landmarks - center
                
                # Normalize relative positions
                max_dist = np.max(np.abs(rel_positions)) + 1e-6
                rel_positions = rel_positions / max_dist
                
                # Calculate pairwise distances between landmarks (each to each)
                # This creates a more robust feature regardless of number of landmarks
                distances = np.zeros((num_landmarks, min(5, num_landmarks)))
                for i in range(num_landmarks):
                    # Calculate distance to all other landmarks
                    all_dists = np.linalg.norm(landmarks - landmarks[i], axis=1)
                    # Sort and take the closest 5 (or fewer if not enough landmarks)
                    closest = np.argsort(all_dists)[1:1+min(5, num_landmarks-1)]
                    distances[i, :len(closest)] = all_dists[closest]
                
                # Normalize distances
                distances = (distances - distances.mean()) / (distances.std() + 1e-6)
                
                # Combine original coordinates with distances and relative positions
                additional_features = np.hstack([distances, rel_positions])
                
                # Concatenate with original features
                x = torch.cat([x, torch.tensor(additional_features, dtype=torch.float)], dim=1)
            
            # Convert edges to edge_index format (Shape: 2, num_edges)
            if edges:
                edge_index = torch.tensor([[i, j] for i, j in edges], dtype=torch.long).t()
                
                # Create edge weights based on distances between connected landmarks
                i_indices, j_indices = edge_index
                edge_weights = torch.norm(x[i_indices, :2] - x[j_indices, :2], dim=1)
                # Normalize edge weights
                edge_weights = 1.0 / (edge_weights + 1e-6)
            else:
                # If no edges were found, create self-loops as a fallback
                edge_index = torch.tensor([[i, i] for i in range(num_landmarks)], dtype=torch.long).t()
                edge_weights = torch.ones(edge_index.shape[1], dtype=torch.float)
            
            # Get the label for this sample
            y = torch.tensor(self.y.iloc[idx], dtype=torch.long)
            
            # Calculate adjacency matrix features to be used globally
            adj_features = torch.tensor(normalized_adjacency, dtype=torch.float)
            
            return Data(
                x=x, 
                edge_index=edge_index, 
                edge_attr=edge_weights,
                y=y, 
                adj=adj_features,
                num_landmarks=torch.tensor(num_landmarks, dtype=torch.long)
            )
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a simple fallback sample if processing fails
            # This ensures the dataset doesn't crash even if a few samples are problematic
            return self._create_fallback_sample(idx)

# Prepare data
X = df['Landmarks']
y = df['encoded_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create datasets with enhanced features
train_dataset = LandmarkDataset(X_train, y_train, add_features=True)
test_dataset = LandmarkDataset(X_test, y_test, add_features=True)

# Calculate the input feature dimension from the dataset
sample_data = train_dataset[0]
num_features = sample_data.x.shape[1]
print(f"Number of node features: {num_features}")

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Multi-layer Graph Attention Network model
class EnhancedGAT(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=128):
        super(EnhancedGAT, self).__init__()
        
        # Graph Attention layers with multi-head attention
        self.conv1 = GATConv(num_features, hidden_channels, heads=4, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels * 2, heads=2, dropout=0.3)
        self.conv3 = GATConv(hidden_channels * 2 * 2, hidden_channels, heads=1, dropout=0.3)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_channels * 4)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2 * 2)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Global attention pooling
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # Region-aware attention pooling
        self.region_attention = nn.ModuleDict({
            region: nn.Sequential(
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            ) for region in FACIAL_REGIONS
        })
        
        # MLP for classification
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch, adj=None):
        # First Graph Attention layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second Graph Attention layer
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Third Graph Attention layer
        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.bn3(x)
        
        # Global mean and max pooling
        x_global_mean = global_mean_pool(x, batch)
        x_global_max = global_max_pool(x, batch)
        
        # Combine global features
        x_global = torch.cat([x_global_mean, x_global_max], dim=1)
        
        # Fully connected layers with residual connection
        x = self.fc1(x_global)
        x = F.elu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

# Define the dual-stream model with adaptive architecture
class AdaptiveDualStreamGNN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=128):
        super(AdaptiveDualStreamGNN, self).__init__()
        
        # GCN stream
        self.gcn1 = GCNConv(num_features, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        
        # GAT stream
        self.gat1 = GATConv(num_features, hidden_channels // 4, heads=4, dropout=0.3)
        self.gat2 = GATConv(hidden_channels, hidden_channels // 2, heads=2, dropout=0.3)
        
        # Batch normalization
        self.bn_gcn1 = nn.BatchNorm1d(hidden_channels)
        self.bn_gcn2 = nn.BatchNorm1d(hidden_channels)
        self.bn_gat1 = nn.BatchNorm1d(hidden_channels)
        self.bn_gat2 = nn.BatchNorm1d(hidden_channels)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_channels * 4, hidden_channels * 2)
        self.bn_fusion = nn.BatchNorm1d(hidden_channels * 2)
        
        # Output layer
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch, adj=None, num_landmarks=None):
        # Apply dropout to input features for regularization
        x = F.dropout(x, p=0.2, training=self.training)
        
        # GCN stream
        x_gcn = self.gcn1(x, edge_index)
        x_gcn = F.relu(x_gcn)
        x_gcn = self.bn_gcn1(x_gcn)
        x_gcn = F.dropout(x_gcn, p=0.3, training=self.training)
        
        x_gcn = self.gcn2(x_gcn, edge_index)
        x_gcn = F.relu(x_gcn)
        x_gcn = self.bn_gcn2(x_gcn)
        
        # GAT stream
        x_gat = self.gat1(x, edge_index)
        x_gat = F.relu(x_gat)
        x_gat = self.bn_gat1(x_gat)
        x_gat = F.dropout(x_gat, p=0.3, training=self.training)
        
        x_gat = self.gat2(x_gat, edge_index)
        x_gat = F.relu(x_gat)
        x_gat = self.bn_gat2(x_gat)
        
        # Global pooling for both streams
        x_gcn_mean = global_mean_pool(x_gcn, batch)
        x_gcn_max = global_max_pool(x_gcn, batch)
        x_gat_mean = global_mean_pool(x_gat, batch)
        x_gat_max = global_max_pool(x_gat, batch)
        
        # Concatenate all global features
        x_global = torch.cat([x_gcn_mean, x_gcn_max, x_gat_mean, x_gat_max], dim=1)
        
        # Feature fusion
        x = self.fusion(x_global)
        x = F.relu(x)
        x = self.bn_fusion(x)
        x = F.dropout(x, p=0.4, training=self.training)
        
        # Classification
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        
        return x

# Define a specialized model for facial landmarks
class FacialLandmarkGNN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=128):
        super(FacialLandmarkGNN, self).__init__()
        
        # Different convolution types for different facial regions
        self.conv_spatial = ChebConv(num_features, hidden_channels, K=3)
        self.conv_structural = GCNConv(hidden_channels, hidden_channels)
        self.conv_attention = GATConv(hidden_channels, hidden_channels, heads=4, dropout=0.3)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels * 4)
        
        # Region-specific feature extraction
        self.region_nets = nn.ModuleDict({
            region: nn.Sequential(
                nn.Linear(hidden_channels * 4, hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for region in FACIAL_REGIONS
        })
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_channels * len(FACIAL_REGIONS), hidden_channels * 2)
        self.bn_fusion = nn.BatchNorm1d(hidden_channels * 2)
        
        # Output layer
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, edge_attr, batch, adj=None):
        # First apply spatial convolution (ChebConv captures more spatial information)
        x = self.conv_spatial(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Then apply structural convolution
        x = self.conv_structural(x, edge_index)
        x = F.relu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Finally apply attention mechanism
        x = self.conv_attention(x, edge_index)
        x = F.relu(x)
        x = self.bn3(x)
        
        # Extract region-specific features
        region_features = []
        batch_size = torch.max(batch) + 1
        
        for region, indices in FACIAL_REGIONS.items():
            # Extract nodes for this region
            mask = torch.zeros(x.size(0), dtype=torch.bool)
            for b in range(batch_size):
                start_idx = b * 68  # Assuming 68 landmarks per face
                end_idx = (b + 1) * 68
                for idx in indices:
                    if start_idx + idx < mask.size(0):
                        mask[start_idx + idx] = True
            
            # Apply region-specific processing if we have nodes for this region
            if torch.any(mask):
                region_x = x[mask]
                region_batch = torch.zeros_like(batch[mask])
                
                # Assign batch indices for pooling
                current_node = 0
                for b in range(batch_size):
                    start_idx = b * 68
                    end_idx = (b + 1) * 68
                    region_size = sum(1 for idx in indices if start_idx + idx < x.size(0))
                    if region_size > 0:
                        region_batch[current_node:current_node + region_size] = b
                        current_node += region_size
                
                # Pool region features
                region_pooled = global_mean_pool(region_x, region_batch)
                
                # Process with region-specific network
                region_processed = self.region_nets[region](region_pooled)
                
                region_features.append(region_processed)
            else:
                # If no nodes for this region (unlikely), add zeros
                region_features.append(torch.zeros(batch_size, hidden_channels, 
                                                  device=x.device))
        
        # Concatenate all region features
        if region_features:
            region_concat = torch.cat(region_features, dim=1)
            
            # Fusion
            x = self.fusion(region_concat)
            x = F.relu(x)
            x = self.bn_fusion(x)
            x = F.dropout(x, p=0.4, training=self.training)
            
            # Classification
            x = self.fc1(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.fc2(x)
            
            return x
        else:
            # Fallback if no region features (shouldn't happen with proper data)
            return torch.zeros(batch_size, len(emotion_list), device=x.device)

# Choose the model to use
# model = EnhancedGAT(num_features, len(emotion_list), hidden_channels=128).to(device)
# model = AdaptiveDualStreamGNN(num_features, len(emotion_list), hidden_channels=128).to(device)
model = FacialLandmarkGNN(num_features, len(emotion_list), hidden_channels=64).to(device)

# Print model summary
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
criterion = nn.CrossEntropyLoss()

# Training loop with training accuracy
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    for batch in loader:
        batch = batch.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.adj)
        
        # Compute loss
        loss = criterion(out, batch.y)
        total_loss += loss.item()
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(out, dim=1)
        correct += (predicted == batch.y).sum().item()
        total += batch.y.size(0)
        
        # Store predictions and labels for confusion matrix later
        all_labels.extend(batch.y.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
    
    train_loss = total_loss / len(loader)
    train_accuracy = correct / total
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return train_loss, train_accuracy, cm

# Testing loop with validation accuracy and loss
def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.adj)
            
            # Compute loss
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            
            # Calculate accuracy
            probs = F.softmax(out, dim=1)
            _, predicted = torch.max(out, dim=1)
            correct += (predicted == batch.y).sum().item()
            total += batch.y.size(0)
            
            # Store predictions and labels for confusion matrix later
            all_labels.extend(batch.y.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss = total_loss / len(loader)
    val_accuracy = correct / total
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return val_loss, val_accuracy, cm, all_labels, all_preds, all_probs

# Function to save the model checkpoint
def save_checkpoint(state, filename="best_model.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

# Training and evaluation
epochs = 100
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
cm_list = []  # List to store confusion matrices
best_val_acc = 0.0

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Train
    train_loss, train_accuracy, cm_train = train(model, train_loader, optimizer, criterion, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validate
    val_loss, val_accuracy, cm_val, val_labels, val_preds, val_probs = test(model, test_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    cm_list.append((cm_train, cm_val))
    
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
        }, filename="best_facial_landmark_gnn.pth")
    
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")
    
    # Early stopping check
    if epoch >= 20 and all(val_accuracies[-5] >= val_accuracies[-5+i] for i in range(1, 5)):
        print("Early stopping triggered. No improvement in validation accuracy for 5 epochs.")
        break

# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot accuracy curve
plt.figure(figsize=(10, 5))
plt.plot(range(len(train_accuracies)), train_accuracies, label='Train Accuracy')
plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig('accuracy_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Load best model for final evaluation
checkpoint = torch.load("best_facial_landmark_gnn.pth")
model.load_state_dict(checkpoint['state_dict'])
print(f"Loaded best model with validation accuracy: {checkpoint['accuracy']*100:.2f}%")

# Final evaluation on test set
final_val_loss, final_val_accuracy, final_cm, test_labels, test_preds, test_probs = test(model, test_loader, criterion, device)
print(f"Final Test Accuracy: {final_val_accuracy*100:.2f}%")

# Display classification report
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=emotion_list))

# Confusion matrix visualization
plt.figure(figsize=(10, 8))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_list, yticklabels=emotion_list)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize emotion predictions
def visualize_landmark_predictions(test_dataset, model, indices, device):
    """
    Visualize predictions on facial landmarks for selected samples
    """
    model.eval()
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        # Get sample
        data = test_dataset[idx].to(device)
        
        # Get prediction
        with torch.no_grad():
            out = model(data.x.unsqueeze(0), data.edge_index.unsqueeze(0), 
                       data.edge_attr.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device), 
                       data.adj.unsqueeze(0))
            probs = F.softmax(out, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        
        # Get ground truth
        true_label = data.y.item()
        
        # Get original landmarks
        landmarks = data.x[:, :2].cpu().numpy()
        
        # Plot landmarks
        ax = axes[i]
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', s=10)
        
        # Plot edges
        edge_index = data.edge_index.cpu().numpy()
        for j in range(edge_index.shape[1]):
            src, dst = edge_index[0, j], edge_index[1, j]
            ax.plot([landmarks[src, 0], landmarks[dst, 0]],
                   [landmarks[src, 1], landmarks[dst, 1]], 'gray', alpha=0.5, linewidth=0.5)
        
        # Plot specific facial regions with different colors
        colors = {
            'jaw': 'green',
            'right_eye': 'red',
            'left_eye': 'red',
            'right_eyebrow': 'purple',
            'left_eyebrow': 'purple',
            'nose_bridge': 'orange',
            'nose_tip': 'orange',
            'outer_lips': 'magenta',
            'inner_lips': 'pink'
        }
        
        for region, indices in FACIAL_REGIONS.items():
            region_landmarks = landmarks[indices]
            ax.scatter(region_landmarks[:, 0], region_landmarks[:, 1], 
                      c=colors[region], s=30, label=region)
        
        # Flip y-axis for correct facial orientation
        ax.invert_yaxis()
        
        # Set title with prediction and ground truth
        ax.set_title(f"True: {emotion_list[true_label]}, Pred: {emotion_list[pred]}")
        
        # Remove axes for cleaner visualization
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add legend to the last subplot
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('landmark_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize some predictions
random_indices = np.random.choice(len(test_dataset), 6, replace=False)
visualize_landmark_predictions(test_dataset, model, random_indices, device)

# Analyze which facial regions contribute most to emotion recognition
def analyze_facial_regions(test_dataset, model, device, num_samples=50):
    """
    Analyze which facial regions contribute most to emotion recognition by masking them out
    """
    model.eval()
    
    # Get random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    # Dictionary to store importance of each region
    region_importance = {region: [] for region in FACIAL_REGIONS}
    
    for idx in indices:
        # Get sample
        data = test_dataset[idx].to(device)
        
        # Get original prediction
        with torch.no_grad():
            out_original = model(data.x.unsqueeze(0), data.edge_index.unsqueeze(0), 
                               data.edge_attr.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device),
                               data.adj.unsqueeze(0))
            probs_original = F.softmax(out_original, dim=1)
            pred_original = torch.argmax(probs_original, dim=1).item()
            confidence_original = probs_original[0, pred_original].item()
        
        # True label
        true_label = data.y.item()
        
        # Skip samples where original prediction is wrong
        if pred_original != true_label:
            continue
        
        # For each region, mask it out and see how prediction changes
        for region, indices in FACIAL_REGIONS.items():
            # Create a copy of the features
            x_masked = data.x.clone()
            
            # Mask the region by setting features to mean values
            x_masked[indices, :] = x_masked.mean(dim=0)
            
            # Get prediction with masked features
            with torch.no_grad():
                out_masked = model(x_masked.unsqueeze(0), data.edge_index.unsqueeze(0), 
                                 data.edge_attr.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device),
                                 data.adj.unsqueeze(0))
                probs_masked = F.softmax(out_masked, dim=1)
                confidence_masked = probs_masked[0, true_label].item()
            
            # Calculate importance as drop in confidence
            importance = confidence_original - confidence_masked
            region_importance[region].append(importance)
    
    # Calculate mean importance for each region
    mean_importance = {region: np.mean(values) for region, values in region_importance.items()}
    
    # Plot region importance
    plt.figure(figsize=(12, 6))
    regions = list(mean_importance.keys())
    importances = list(mean_importance.values())
    
    # Sort by importance
    sorted_indices = np.argsort(importances)[::-1]
    regions = [regions[i] for i in sorted_indices]
    importances = [importances[i] for i in sorted_indices]
    
    plt.bar(regions, importances, color='skyblue')
    plt.xlabel('Facial Region')
    plt.ylabel('Importance (Confidence Drop)')
    plt.title('Importance of Facial Regions for Emotion Recognition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('region_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean_importance

# Analyze facial region importance
region_importance = analyze_facial_regions(test_dataset, model, device)
print("Facial Region Importance:")
for region, importance in sorted(region_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{region}: {importance:.4f}")

# Analyze model performance by emotion
def analyze_emotion_performance(test_labels, test_preds, emotion_list):
    """
    Analyze model performance for each emotion
    """
    # Calculate accuracy for each emotion
    emotion_acc = {}
    emotion_counts = {}
    
    for true, pred in zip(test_labels, test_preds):
        emotion = emotion_list[true]
        if emotion not in emotion_counts:
            emotion_counts[emotion] = 0
            emotion_acc[emotion] = 0
        
        emotion_counts[emotion] += 1
        if true == pred:
            emotion_acc[emotion] += 1
    
    # Calculate accuracy percentage
    for emotion in emotion_acc:
        emotion_acc[emotion] /= emotion_counts[emotion]
    
    # Plot emotion accuracy
    plt.figure(figsize=(12, 6))
    emotions = list(emotion_acc.keys())
    accuracies = list(emotion_acc.values())
    counts = [emotion_counts[e] for e in emotions]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    emotions = [emotions[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    # Create bar chart with count annotations
    bars = plt.bar(emotions, [acc*100 for acc in accuracies], color='skyblue')
    plt.xlabel('Emotion')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy by Emotion')
    plt.xticks(rotation=45, ha='right')
    
    # Annotate with counts
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, 5, 
                f"n={counts[i]}", 
                ha='center', va='bottom', color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('emotion_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return emotion_acc, emotion_counts

# Analyze emotion performance
emotion_acc, emotion_counts = analyze_emotion_performance(test_labels, test_preds, emotion_list)
print("\nEmotion-wise Accuracy:")
for emotion, acc in sorted(emotion_acc.items(), key=lambda x: x[1], reverse=True):
    print(f"{emotion}: {acc*100:.2f}% (n={emotion_counts[emotion]})")

# Save the final model
torch.save({
    'model': model.state_dict(),
    'emotion_list': emotion_list,
    'encoder': encoder,
    'config': {
        'num_features': num_features,
        'num_classes': len(emotion_list),
        'hidden_channels': 128
    }
}, 'facial_emotion_gnn_final.pth')

print("Final model saved as 'facial_emotion_gnn_final.pth'")
