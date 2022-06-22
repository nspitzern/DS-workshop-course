import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 10
LR = 7e-4
CRITERION = nn.MSELoss
HIDDEN_SIZE = 100
OPTIMIZER = optim.Adam
BATCH_SIZE = 1000
DROPOUT = 0.3

class SpotifyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop=0.3):
        super().__init__()
        
        self.ff = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(drop),
            nn.SELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        out = self.ff(x)
        return out
    
def get_dnn_model(in_dim):
    return Model(in_dim, HIDDEN_SIZE, 1).to(device)

def get_dnn_results(X_train, X_test, y_train, y_test, nn_model, verbose=True):
    train_data = torch.tensor(X_train.values).float().to(device)
    train_labels = torch.tensor(y_train.values).float().to(device)
    train_dataset = SpotifyDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    test_data = torch.tensor(X_test.values).float().to(device)
    test_labels = torch.tensor(y_test.values).float().to(device)
    test_dataset = SpotifyDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    
    optimizer = OPTIMIZER(nn_model.parameters(), lr=LR)
    criterion = CRITERION()
    
    train_losses, test_losses = _train_dnn(nn_model, train_loader, test_loader, len(train_dataset), len(test_dataset), optimizer, criterion, EPOCHS, device, verbose)
    
    return train_losses, test_losses
    
def _train_dnn(nn_model, train_loader, test_loader, train_data_size, test_data_size, optimizer, criterion, epochs, device, verbose=True):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        nn_model.train()
        train_loss = 0

        for idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            preds = nn_model(data)

            loss = criterion(preds, labels)
            train_loss += loss.item()

            loss.backward()

            optimizer.step()

        
        train_loss = train_loss / (train_data_size * BATCH_SIZE)
        
        test_loss = _evaluate_dnn(nn_model, test_loader, test_data_size, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if verbose:
            print(f'[{epoch + 1}/{EPOCHS}] MSE loss train: {train_loss}, MSE loss test: {test_loss}')
        
    return train_losses, test_losses
    

def _evaluate_dnn(nn_model, data_loader, test_data_size, criterion, device):
    nn_model.eval()

    with torch.no_grad():
        total_loss = 0

        for idx, (data, labels) in enumerate(data_loader):
            data = data.to(device)
            labels = labels.to(device)

            preds = nn_model(data)

            loss = criterion(preds, labels)

            total_loss += loss.item()

        total_loss = total_loss / data_size
        return total_loss
    
def dnn_predict(nn_model, X_test, y_test, device):
    test_data = torch.tensor(X_test.values).float().to(device)
    test_labels = torch.tensor(y_test.values).float().to(device)
    
    y_hat = nn_model(test_data)
    
    return y_hat.detach().numpy()