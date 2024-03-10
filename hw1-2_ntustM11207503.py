import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 創建神經網路模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

# 輸入數據、訓練期望值
x_train = torch.FloatTensor([[0, 0], [1, 1], [0, 1], [1, 0]])
y_train = torch.FloatTensor([[0], [1], [0], [1]])

# 定義模型、損失函數和優化器
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 訓練模型
epochs = 80
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
# 繪製曲線
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Convergence')
plt.show()

# 權重、輸出
print("Weights of the network:")
print(model.fc.weight)

with torch.no_grad():
    output_00 = model(torch.FloatTensor([[0, 0]]))
    output_11 = model(torch.FloatTensor([[1, 1]]))
    output_01 = model(torch.FloatTensor([[0, 1]]))
    output_10 = model(torch.FloatTensor([[1, 0]]))
print("Output for (0, 0):", output_00.item())
print("Output for (1, 1):", output_11.item())
print("Output for (0, 1):", output_01.item())
print("Output for (1, 0):", output_10.item())
