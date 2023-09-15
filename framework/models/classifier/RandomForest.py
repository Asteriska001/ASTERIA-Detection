import torch
from sklearn.ensemble import RandomForestClassifier as CPU_RandomForest
from cuml.ensemble import RandomForestClassifier as GPU_RandomForest

class RandomForest:

    def __init__(self, device='cpu', n_estimators=100, max_depth=None):
        self.device = device

        if self.device == 'cuda':
            self.model = GPU_RandomForest(n_estimators=n_estimators, max_depth=max_depth)
        else:
            self.model = CPU_RandomForest(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        if self.device == 'cuda':
            X = torch.tensor(X).cuda().contiguous()
            y = torch.tensor(y).cuda().contiguous()
        
        self.model.fit(X, y)

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if self.device == 'cuda':
            X = torch.tensor(X).cuda().contiguous()

        preds = self.model.predict(X)
        return torch.tensor(preds).to(self.device)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred.cpu().numpy())

if __name__ == '__main__':
    # 示例
    X_train = torch.rand((100, 30))
    y_train = (X_train.sum(dim=1) > 15).long()
    X_test = torch.rand((20, 30))
    y_test = (X_test.sum(dim=1) > 15).long()

    # 使用CPU模型
    rf_cpu = RandomForest(device='cpu')
    rf_cpu.fit(X_train, y_train)
    accuracy_cpu = rf_cpu.evaluate(X_test, y_test)
    print(f"CPU Accuracy: {accuracy_cpu:.4f}")

    # 使用GPU模型（如果可用）
    if torch.cuda.is_available():
        rf_gpu = RandomForest(device='cuda')
        rf_gpu.fit(X_train, y_train)
        accuracy_gpu = rf_gpu.evaluate(X_test, y_test)
        print(f"GPU Accuracy: {accuracy_gpu:.4f}")
