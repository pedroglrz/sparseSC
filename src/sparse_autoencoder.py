import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from tqdm import tqdm

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lambda_l1=0.001, device='auto', verbose=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lambda_l1 = lambda_l1
        self.verbose = verbose
        self.set_device(device)

        # Encoder weights and bias
        self.W_e = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.b_e = nn.Parameter(torch.zeros(hidden_dim))
        
        # Decoder weights and bias
        self.W_d = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.b_d = nn.Parameter(torch.zeros(input_dim))
        
        # Initialize weights using Kaiming Uniform
        nn.init.kaiming_uniform_(self.W_e)
        nn.init.kaiming_uniform_(self.W_d)

        # Normalize columns to unit norm
        self.W_e.data = self.normalize_columns(self.W_e.data)
        self.W_d.data = self.normalize_columns(self.W_d.data)

    def fit(self, dataloaders, lr= .001, save_dir=None, n_epochs=1500, patience=10):
        # Set run attributes
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.save_dir = save_dir
        self.n_epochs = n_epochs
        self.patience = patience

        # Move to device
        self.to(self.device)
        self.patience_counter = 0
        avg_train_losses = []
        avg_val_losses = []
        avg_train_sparcity = []
        avg_val_sparcity = []        
        self.best_val_loss = float('inf')
        self.best_model_state = None

        for epoch in tqdm(range(n_epochs), desc="Training Progress"):
            training_loss, training_sparcity = self.run_training_loop(dataloaders['train'])
            validation_loss, validation_sparcity = self.run_validation_loop(dataloaders['val'])

            avg_train_losses.append(training_loss / len(dataloaders['train']))
            avg_val_losses.append(validation_loss / len(dataloaders['val']))
            avg_train_sparcity.append(training_sparcity/ len(dataloaders['train']))
            avg_val_sparcity.append(validation_sparcity/ len(dataloaders['val']))
            self.record_loss(epoch, avg_train_losses[-1], avg_val_losses[-1])
            
            self.cache_best_model(avg_val_losses)
            if self.check_stop_condition():
                print(f"Early stopping triggered after {len(avg_val_losses)} epochs")
                break
        
        # Restore the best model state
        if self.best_model_state is not None:
            self.load_state_dict(self.best_model_state)

        # Save the best model and losses at the end of training
        if self.save_dir is not None:
            self.save_results(avg_train_losses, avg_val_losses,avg_train_sparcity, avg_val_sparcity)

    def transform(self, data):
        self.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():
            data = data.to(self.device)
            x, f = self(data)
        
        return x, f

    # Core Model Methods
    def forward(self, x):
        W_e_normalized = self.normalize_columns(self.W_e)
        W_d_normalized = self.normalize_columns(self.W_d)

        x_bar = x - self.b_d
        f = torch.relu(torch.matmul(W_e_normalized, x_bar.T).T + self.b_e)
        x_hat = torch.matmul(W_d_normalized, f.T).T + self.b_d
        
        return x_hat, f
    
    @staticmethod
    def normalize_columns(weight_matrix):
        return weight_matrix / torch.norm(weight_matrix, dim=0, keepdim=True)

    def loss_function(self, x, x_hat, f):
        mse_loss = torch.mean((x - x_hat) ** 2)
        l1_loss = self.lambda_l1 * torch.sum(torch.abs(f))
        loss = mse_loss + l1_loss
        return loss
    
    def apply_orthogonal_gradient(self):
        with torch.no_grad():
            for param in [self.W_e, self.W_d]:
                # Normalize columns of the weight matrices to maintain unit norm
                param_normalized = self.normalize_columns(param.data)
                
                # Compute the gradient projection
                grad_parallel = torch.sum(param.grad * param_normalized, dim=0, keepdim=True) * param_normalized
                
                # Subtract the parallel component from the gradient to keep it orthogonal
                param.grad -= grad_parallel

    # Training Helper Methods
    def run_training_loop(self, dataloader):
        self.train()        
        training_loss = 0
        training_sparsity = 0
        for train_batch in dataloader:
            # Forward pass
            train_batch = train_batch[0].to(self.device)
            self.optimizer.zero_grad()
            x_hat, f = self(train_batch)

            # Calculate loss
            training_batch_loss = self.loss_function(train_batch, x_hat, f)
            training_loss += training_batch_loss.item()

            # Backward pass
            training_batch_loss.backward()
            self.apply_orthogonal_gradient()
            self.optimizer.step()                 

            # Calculate sparcity (fraction of zeros in the hidden layer)
            train_batch_sparsity = (f == 0).float().mean().item()
            training_sparsity += train_batch_sparsity
        
        return training_loss, training_sparsity
    
    def run_validation_loop(self, dataloader):
        self.eval()
        validation_loss = 0.0
        validation_sparsity = 0.0
        with torch.no_grad():
            for val_batch in dataloader:
                # Forward pass
                val_batch = val_batch[0].to(self.device)
                x_hat, f = self(val_batch)

                #calculate loss
                val_batch_loss = self.loss_function(val_batch, x_hat, f)
                validation_loss += val_batch_loss.item()

                #calculate sparcity (fraction of zeros in the hidden layer)
                val_batch_sparsity = (f == 0).float().mean().item()
                validation_sparsity += val_batch_sparsity
        
        return validation_loss, validation_sparsity


    # Utility and Logging Methods
    def set_device(self, device):
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
                
    def record_loss(self, epoch, avg_train_loss, avg_val_loss):  
        log_message = f'Epoch [{epoch+1}/{self.n_epochs}], Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}'
        
        if self.verbose:
            print(log_message)

        if self.save_dir is not None:        
            log_file_path = os.path.join(self.save_dir, 'log.txt')
            os.makedirs(self.save_dir, exist_ok=True)
            with open(log_file_path, 'a') as log_file:
                log_file.write(log_message + '\n')

    def cache_best_model(self, avg_val_losses):
        if avg_val_losses[-1] < self.best_val_loss:
            self.best_val_loss = avg_val_losses[-1]
            self.best_model_state = self.state_dict()
            self.patience_counter = 0  # Reset the patience counter when a new best is found
            if self.verbose:
                print(f"New best model found at epoch {len(avg_val_losses)} with validation loss: {self.best_val_loss}")
        else:
            self.patience_counter += 1
    
    def check_stop_condition(self):
        if self.patience_counter >= self.patience:
            return True
        return False

    
    def save_results(self, avg_train_losses, avg_val_losses, avg_train_sparcity, avg_val_sparcity):
        print(f"Saving best model to {self.save_dir}")
        torch.save(self.state_dict(), os.path.join(self.save_dir, "best_model.pth"))
        
        print(f"Saving average losses")
        losses_df = pd.DataFrame({
            'avg_train_losses': avg_train_losses,
            'avg_val_losses': avg_val_losses,
            'avg_train_sparcity': avg_train_sparcity,
            'avg_vale_sparcity': avg_val_sparcity,

        })
        losses_df.to_csv(os.path.join(self.save_dir, "losses.csv"), index=False)