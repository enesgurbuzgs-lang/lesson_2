from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from utils import make_classification_data, accuracy, log_epoch, ClassificationDataset

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class LogisticRegressionManual:
    def __init__(self, in_features):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)

    def __call__(self, X):
        return sigmoid(X @ self.w + self.b)

    def parameters(self):
        return [self.w, self.b]

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        error = y_pred - y
        self.dw = (X.T @ error) / n
        self.db = error.mean(0)

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def save(self, path):
        torch.save({'w': self.w, 'b': self.b}, path)

    def load(self, path):
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', choices=('random', 'breast_cancer'), default='random')
    args = parser.parse_args()
    X, y = make_classification_data(n=200, source=args.source)
    if args.source != 'random':
        # Приводим признаки реального набора к сопоставимому масштабу.
        X = (X - X.mean(dim=0)) / X.std(dim=0, unbiased=False).clamp_min(1e-8)
    in_features = X.shape[1]
    
    # Создаём датасет и даталоадер
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    print(f'Пример данных: {dataset[0]}')
    
    # Обучаем модель
    model = LogisticRegressionManual(in_features=in_features)
    lr = 0.1
    epochs = 100
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            y_pred = model(batch_X)
            loss = -(batch_y * torch.log(y_pred + 1e-8) + (1 - batch_y) * torch.log(1 - y_pred + 1e-8)).mean().item()
            acc = accuracy(y_pred, batch_y)
            
            total_loss += loss
            total_acc += acc
            
            model.zero_grad()
            model.backward(batch_X, batch_y, y_pred)
            model.step(lr)
        
        avg_loss = total_loss / (i + 1)
        avg_acc = total_acc / (i + 1)
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc)
    
    model.save('logreg_manual.pth') 