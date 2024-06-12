
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Set manual seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

class Autoencoder(nn.Module):
    def __init__(self, input_dim=288*4, output_dim=288, dropout_rate=0):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 288*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(288*2, 288),
            nn.ReLU(),
            # nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            # nn.Linear(288, 144),
            # nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        self.decoder = nn.Sequential(
            nn.Linear(288, output_dim),
            nn.Sigmoid()
        )

        # Initialize weights similar to Keras
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Keras uses glorot_uniform by default
            nn.init.zeros_(m.bias)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def custom_loss(y_true, y_pred):
    mask = (y_true != 0).float()
    masked_squared_error = (y_true * mask - y_pred * mask) ** 2
    loss = torch.sum(masked_squared_error) / torch.sum(mask)
    return loss

def autoencoder_pytorch(X_train_autoencoder, y_train_autoencoder, num_epochs=70, batch_size=64):
    input_dim = 288 * 4
    output_dim = 288
    dropout_rate = 0.1  

    model = Autoencoder(input_dim=input_dim, output_dim=output_dim, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Adjusted learning rate

    X_train_autoencoder = torch.tensor(X_train_autoencoder, dtype=torch.float32)
    y_train_autoencoder = torch.tensor(y_train_autoencoder, dtype=torch.float32)

    # Training loop
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X_train_autoencoder)
        
        # Compute custom loss
        loss = custom_loss(y_train_autoencoder, output)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
    
    return model



def create_autoencoder(dropout_rate=0):
    from keras.optimizers import Adam
    input_dim = 288 * 4
    output_dim = 288

    # Define the denoising autoencoder model
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(288*2, activation='relu')(input_layer)
    if dropout_rate>0:
        encoded = Dropout(dropout_rate)(encoded)  
    encoded = Dense(432, activation='relu')(encoded)
    if dropout_rate>0:
        encoded = Dropout(dropout_rate)(encoded)      
    # encoded = Dense(144, activation='relu')(encoded)
    # if dropout_rate>0:
    #     encoded = Dropout(dropout_rate)(encoded)  
    decoded = Dense(output_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    
    def custom_loss(y_true, y_pred):
        # Mask for excluding certain locations
        # we consider 0 as a missing data
        mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
        # Compute squared error
        masked_squared_error = K.square(y_true*mask - y_pred*mask)
        loss = K.sum(masked_squared_error) / (K.sum(mask))
        return loss
    
    autoencoder.compile(optimizer="Adam", loss=custom_loss)

    return autoencoder


def create_autoencoder_simple(dropout_rate=0):
    input_dim = 288 * 4
    output_dim = 288

    # Define the denoising autoencoder model
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(288*2, activation='relu')(input_layer)
    if dropout_rate>0:
        encoded = Dropout(dropout_rate)(encoded)  
    decoded = Dense(output_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    
    def custom_loss(y_true, y_pred):
        # Mask for excluding certain locations
        # we consider 0 as a missing data
        mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
        # Compute squared error
        squared_error = K.square(y_true - y_pred)
        # Apply the mask to exclude specific locations
        masked_squared_error = squared_error * mask
        # Compute the mean loss excluding the masked locations
        loss = K.sum(masked_squared_error) / (K.sum(mask))
        return loss
    
    autoencoder.compile(optimizer="adam", loss=custom_loss)

    return autoencoder


