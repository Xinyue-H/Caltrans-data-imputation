import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_heads, latent_dim):
        super(GraphAttentionNetwork, self).__init__()
        self.gat = GATConv(in_channels, hidden_dim, heads=num_heads)
        self.encoder = nn.Linear(hidden_dim * num_heads, latent_dim)
        self.decoder = nn.Linear(latent_dim, hidden_dim * num_heads)
        self.output_layer = nn.Linear(hidden_dim * num_heads, in_channels)
        
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = F.elu(x)
        
        # Autoencoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        x = F.elu(decoded)
        x = self.output_layer(x)
        return x
    
def train_GAT(A, X_train, X_val, mask_train, mask_val, num_epochs=70):
    # Initialize model
    in_channels = 288
    hidden_dim = 144  # Hidden dimension for GAT layer
    num_heads = 3     # Number of attention heads
    latent_dim = 288  # Latent dimension for autoencoder

    model = GraphAttentionNetwork(in_channels=in_channels, hidden_dim=hidden_dim, num_heads=num_heads, latent_dim=latent_dim)

    # Convert adjacency matrix to edge index format for torch_geometric
    edge_index = torch.nonzero(A).t()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0004)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)  # Decay LR by a factor of 0.2 every 10 epochs

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Training
        for t in range(X_train.shape[0]):
            optimizer.zero_grad()
            
            # Get the features for time step t
            features = X_train[t]
            mask = mask_train[t]
            
            # Forward pass
            outputs = model(features, edge_index)
            
            # Compute loss only on known values
            loss = criterion(outputs * mask, features * mask)  # Apply mask to consider only known values

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for t in range(X_val.shape[0]):
                features = X_val[t]
                mask = mask_val[t]
                
                outputs = model(features, edge_index)
                loss = criterion(outputs * mask, features * mask)
                val_loss += loss.item()
        if epoch%10==0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / X_train.shape[0]}, Validation Loss: {val_loss / X_val.shape[0]}')

    print("Training complete.")
    return model