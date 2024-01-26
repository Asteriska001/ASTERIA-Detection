import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score

class VulDeepecker(nn.Module):
    def __init__(self, 
                 #vectors, 
                 #labels, 
                 name="", batch_size=64, **kwargs):
        super(VulDeepecker, self).__init__()
        '''
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
        '''
        
        self.lstm = nn.LSTM(input_size= 50,#400,vectors.shape[2], 
                            hidden_size=300, 
                            batch_first=True, 
                            bidirectional=True)
        self.fc1 = nn.Linear(600, 300)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(300, 300)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(300, 2)
    
    def forward(self, x):
        print('vuldeepcker input shape:', x.shape)
        x = x.float()
        #input_x = x.unsqueeze(1).float()
        x, _ = self.lstm(x)
        x = x[:, -1, :] # 选择最后一个时间步长的输出
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.softmax(self.fc3(x), dim=-1)
        return x
    '''
    def forward(self, x):
        #author added.
        seq_len = 1 # 你可以根据你的序列长度设置这个值
        # 在序列维度上添加一个新维度
        input_x = x.unsqueeze(1) # 现在input_x的形状为(batch_size, 1, input_size)
        input_x = input_x.float()
        x = input_x
        
        x, _ = self.lstm(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.softmax(self.fc3(x), dim=-1)
        return x
    '''
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
