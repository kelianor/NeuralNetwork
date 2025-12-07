import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Аугментация: небольшие сдвиги, повороты, масштабирование
transform = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # превращаем 28x28 в вектор
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
for epoch in range(10):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, loss={loss.item():.4f}")

# Сохранение весов в текстовые файлы
for name, param in model.named_parameters():
    arr = param.detach().cpu().numpy()
    with open(f"./data/{name}.txt", "w") as f:
        if arr.ndim == 1:
            for row in arr:
                f.write(f"{row}\n")
        else:
            for row in arr:
                f.write(" ".join(map(str, row)) + "\n")
