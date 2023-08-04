import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score

class BLSTM(nn.Module):
    def __init__(self, vectors, labels, name="", batch_size=64):
        super(BLSTM, self).__init__()
        self.name = name
        self.batch_size = batch_size

        positive_idxs = labels == 1
        negative_idxs = labels == 0
        undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=False)
        resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])

        X_train, X_test, y_train, y_test = train_test_split(vectors[resampled_idxs], labels[resampled_idxs],
                                                            test_size=0.2, stratify=labels[resampled_idxs])
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)
        
        self.lstm = nn.LSTM(input_size=vectors.shape[2], hidden_size=300, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(600, 300)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(300, 300)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(300, 2)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.softmax(self.fc3(x), dim=-1)
        return x
'''
    def train_model(self):
        dataset = TensorDataset(self.X_train, self.y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adamax(self.parameters(), lr=0.002)

        for epoch in range(4):
            for batch in dataloader:
                x, y = batch
                optimizer.zero_grad()
                outputs = self.forward(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
        torch.save(self.state_dict(), self.name + "_model.pt")

    def test_model(self):
        self.load_state_dict(torch.load(self.name + "_model.pt"))
        with torch.no_grad():
            outputs = self.forward(self.X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(predicted.numpy(), self.y_test.numpy())
            tn, fp, fn, tp = confusion_matrix(self.y_test.numpy(), predicted.numpy()).ravel()
        print('Accuracy is...', accuracy)
        print('False positive rate is...', fp / (fp + tn))
        print('False negative rate is...', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('True positive rate is...', recall)
        precision = tp / (tp + fp)
        print('Precision is...', precision)
        print('F1 score is...', (2 * precision * recall) / (precision + recall))
'''
