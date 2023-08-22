#https://github.com/hazimhanif/svd_exp1/blob/master/SVD_2.ipynb
import torch
import torch.nn as nn
import torch.optim as optim

class Russell(nn.Module):
    def __init__(self, WORDS_SIZE, INPUT_SIZE, random_weights):
        super(Russell, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=WORDS_SIZE, embedding_dim=13)
        self.embedding.weight.data.copy_(torch.from_numpy(random_weights))

        # Convolutional layer
        self.conv1d = nn.Conv1d(in_channels=13, out_channels=512, kernel_size=9, padding='same', stride=1)

        # Max pooling layer
        self.maxpool = nn.MaxPool1d(kernel_size=5)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * (INPUT_SIZE // 5), 64)  # you might need to adjust the input size
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv1d(x.transpose(1, 2))  # Switch the channel with sequence length
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.Dropout(0.5)(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.Sigmoid()(self.fc3(x))
        return x

# Model initialization
'''
random_weights_np = ...  # this is a placeholder. Make sure you have the `random_weights` numpy array here
random_weights_tensor = torch.Tensor(random_weights_np)
model = CNN(WORDS_SIZE, INPUT_SIZE, random_weights_tensor)
print(model)
'''