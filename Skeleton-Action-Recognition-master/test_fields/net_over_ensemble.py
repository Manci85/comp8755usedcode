import torch
import torch.nn as nn


class NetOverEnsemble(nn.Module):
    def __init__(self, in_channel, class_num):
        super().__init__()

        self.conv1d_0 = nn.Conv1d(in_channel, out_channels=16, kernel_size=1)
        self.conv1d_1 = nn.Conv1d(16, out_channels=32, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(32, out_channels=16, kernel_size=1)
        self.conv1d_3 = nn.Conv1d(16, out_channels=1, kernel_size=1)

        self.fc_0 = nn.Linear(class_num, class_num*2)
        self.fc_1 = nn.Linear(class_num*2, class_num*2)
        self.fc_2 = nn.Linear(class_num*2, class_num)

        # self.conv1d_0 = nn.Conv1d(in_channel, out_channels=1, kernel_size=1)
        # self.conv1d_0.weight.data.fill_(1.0)

    def forward(self, x):
        out = torch.relu(self.conv1d_0(x))
        out = torch.relu(self.conv1d_1(out))
        out = torch.relu(self.conv1d_2(out))
        out = torch.relu(self.conv1d_3(out))

        out = torch.relu(self.fc_0(out))
        out = torch.relu(self.fc_1(out))
        out = self.fc_2(out)

        return out

        # return self.conv1d_0(x)