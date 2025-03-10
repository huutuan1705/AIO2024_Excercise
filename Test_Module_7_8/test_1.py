import torch
import torch.nn as nn

net = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2)
flatten = nn.Flatten()
linear = nn.Linear(6, 6)
rnn = nn.RNN(input_size=6, hidden_size=3)



input = torch.randn(1, 2, 4)
# output = net(input)
output = flatten(input)
# output, a = rnn(input)
print(output.shape)
# print(a.shape)