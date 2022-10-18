import torch
import torch.nn as nn
import torch.optim as optim

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# net = Net()
# print(net)

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# # Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in net.state_dict():
#     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

# print()

# # Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])
# fc1 = nn.Linear(784, 500)
# fc2 = nn.Linear(500, 10)
# # optimizer = torch.optim.SGD([fc1.parameters(), fc2.parameters()], lr=0.01) 
# optimizer = torch.optim.SGD(list(fc1.parameters()) + list(fc2.parameters()), lr=0.01)

loss = nn.MSELoss()
input = torch.randn(3, 1, requires_grad=True)
target = torch.randn(3, 1)
output = loss(input, target)
output.backward()

print(input)
print(target)
print(output)