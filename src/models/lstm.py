import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take last time step
        out = self.dropout(lstm_out)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out

class PyTorchLSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape, hidden_size=64, num_layers=2, dropout=0.2,
                 learning_rate=0.001, batch_size=32, epochs=50, device=None):
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.criterion = nn.BCELoss()
    
    def _init_model(self):
        self.model = LSTMNetwork(
            input_size=self.input_shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _prepare_data(self, X, y=None):
        X_tensor = torch.FloatTensor(X.reshape(-1, 1, X.shape[1])).to(self.device)
        if y is not None:
            y_tensor = torch.FloatTensor(y.values.reshape(-1, 1)).to(self.device)
            return torch.utils.data.TensorDataset(X_tensor, y_tensor)
        return X_tensor
    
    def _create_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
    
    def fit(self, X, y):
        if self.model is None:
            self._init_model()

        dataset = self._prepare_data(X, y)
        train_loader = self._create_dataloader(dataset)
        
        self.model.train() # type: ignore
        with Progress() as progress:
            task = progress.add_task("[cyan]Training LSTM...", total=self.epochs)
            
            for epoch in range(self.epochs):
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X) # type: ignore
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
                
                progress.update(task, advance=1)
                    
        return self
    
    def predict_proba(self, X):
        self.model.eval() # type: ignore
        X_tensor = self._prepare_data(X)
        
        with torch.no_grad():
            probas = self.model(X_tensor).cpu().numpy() # type: ignore
        
        return np.hstack([1 - probas, probas])
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)