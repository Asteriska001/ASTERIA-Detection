import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.5):
        super(MLP, self).__init__()

        layers = []
        in_features = input_size
        for out_features in hidden_layers:
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = out_features

        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    class MLPForVulnerabilityDetection:
        def __init__(self, input_size, hidden_layers=[128, 64], output_size=1, dropout_rate=0.5, lr=0.001, weight_decay=1e-5):
            self.model = MLP(input_size, hidden_layers, output_size, dropout_rate).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def train(self, train_loader, valid_loader, epochs=50, early_stopping_rounds=10):
            self.model.to(self.device)

            best_loss = float('inf')
            stopping_counter = 0
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                for data, labels in train_loader:
                    data, labels = data.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(data).squeeze()
                    loss = self.criterion(outputs, labels.float())
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()

                # Validation
                valid_loss = self.evaluate(valid_loader)
                print(f'Epoch {epoch+1}/{epochs} - Train loss: {train_loss/len(train_loader):.4f} - Validation loss: {valid_loss:.4f}')

                # Early stopping
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    stopping_counter = 0
                else:
                    stopping_counter += 1

                if stopping_counter >= early_stopping_rounds:
                    print("Early stopping triggered.")
                    break

        def evaluate(self, loader):
            self.model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for data, labels in loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs = self.model(data).squeeze()
                    loss = self.criterion(outputs, labels.float())
                    total_loss += loss.item()
            return total_loss / len(loader)

        def predict(self, data):
            self.model.eval()
            if not torch.is_tensor(data):
                data = torch.tensor(data)
            data = data.to(self.device)
            with torch.no_grad():
                logits = self.model(data).squeeze()
                probs = torch.sigmoid(logits)
            return probs.cpu()

