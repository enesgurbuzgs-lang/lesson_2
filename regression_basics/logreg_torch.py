from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_classification_data, accuracy, log_epoch, ClassificationDataset

class LogisticRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    # Генерируем данные
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
    
    # Создаём модель, функцию потерь и оптимизатор
    model = LogisticRegression(in_features=in_features)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            # Вычисляем accuracy
            y_pred = torch.sigmoid(logits)
            acc = accuracy(y_pred, batch_y)
            
            total_loss += loss.item()
            total_acc += acc
        
        avg_loss = total_loss / (i + 1)
        avg_acc = total_acc / (i + 1)
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc)
    
    # Сохраняем модель
    torch.save(model.state_dict(), 'logreg_torch.pth')
    
    # Загружаем модель
    new_model = LogisticRegression(in_features=in_features)
    new_model.load_state_dict(torch.load('logreg_torch.pth'))
    new_model.eval() 