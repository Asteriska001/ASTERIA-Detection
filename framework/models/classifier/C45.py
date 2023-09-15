import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class C45:
    def __init__(self, max_depth=None, balanced=False):
        class_weight = 'balanced' if balanced else None
        self.model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, class_weight=class_weight)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        self.model.fit(X, y)

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        preds = self.model.predict(X)
        return torch.tensor(preds).to(self.device)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        return accuracy_score(y_true, y_pred.cpu().numpy())

if __name__ == '__main__':
    # 示例
    X_train = torch.rand((100, 30))
    y_train = (X_train.sum(dim=1) > 15).long()
    X_test = torch.rand((20, 30))
    y_test = (X_test.sum(dim=1) > 15).long()

    classifier = C45(balanced=True)
    classifier.fit(X_train, y_train)
    accuracy = classifier.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")
