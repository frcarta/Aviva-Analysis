import torch.nn as nn
import torch.nn.functional as func
import torch as torch


class R_PNN_model(nn.Module):
    def __init__(self, padding="valid", scope=6, bias=True) -> None:
        super(R_PNN_model, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 7, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(32, 48, 5, padding=padding, bias=bias)
        self.conv3 = nn.Conv2d(48, 1, 3, padding=padding, bias=bias)

        self.scope = scope

    # input is padded with scope, which should be set as a parameter, even though here it is fixed, it is the net_scope parameter in test.py
    def forward(self, input):
        x = func.relu(self.conv1(input))
        x = func.relu(self.conv2(x))  # + self.mp1(input[:, 0, :, :])
        # x = self.conv3(x) + self.mp2(x1)
        x = self.conv3(x)  # + self.mp2(input[:, 0, :, :])
        # x = x + input[:, :-1, self.scope : -self.scope, self.scope : -self.scope]
        return x


##----------------------------------------------------------------------------------------------------------------------------------------------
class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3,
            dilation=1,
            bias=self.bias,
        )

    def forward(self, x):
        max = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        concat = torch.cat((max, avg), dim=1)
        output = self.conv(concat)
        output = func.sigmoid(output) * x
        return output


class R_PNN_model3x3(nn.Module):
    def __init__(self, padding="valid", scope=6, bias=True) -> None:
        super(R_PNN_model3x3, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(32, 48, 3, padding=padding, bias=bias)
        self.conv3 = nn.Conv2d(48, 1, 3, padding=padding, bias=bias)

        self.att_layer = SAM()

        self.mp1 = nn.AdaptiveAvgPool2d((108, 138))
        self.mp2 = nn.AdaptiveAvgPool2d((106, 136))

        self.scope = scope

    # input is padded with scope, which should be set as a parameter, even though here it is fixed, it is the net_scope parameter in test.py
    def forward(self, input):
        x = func.relu(self.conv1(input))
        x = func.relu(self.conv2(x))  # + self.mp1(input[:, 0, :, :])
        # x = self.conv3(x) + self.mp2(x1)
        x = self.conv3(x)  # + self.mp2(input[:, 0, :, :])

        # x = x + input[:, :-1, self.scope : -self.scope, self.scope : -self.scope]
        """
        x = x + self.att_layer(
            input[:, :-1, self.scope : -self.scope, self.scope : -self.scope]
        )

        x = x + input[:, :-1, self.scope : -self.scope, self.scope : -self.scope]
        x = self.att_layer(x)
        """
        return x


class R_PNN_model3x3_res(nn.Module):
    def __init__(self, padding="valid", scope=6, bias=True) -> None:  # scope is given
        super(R_PNN_model3x3_res, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(32, 48, 3, padding=padding, bias=bias)
        self.conv3 = nn.Conv2d(48, 1, 3, padding=padding, bias=bias)

        self.att_layer = SAM()

        self.mp1 = nn.AdaptiveAvgPool2d((108, 138))
        self.mp2 = nn.AdaptiveAvgPool2d((106, 136))

        self.scope = scope

    # input is padded with scope, which should be set as a parameter, even though here it is fixed, it is the net_scope parameter in test.py
    def forward(self, input):
        x = func.relu(self.conv1(input))
        x = func.relu(self.conv2(x))
        x = self.conv3(x)
        x = x + input[:, :-1, self.scope : -self.scope, self.scope : -self.scope]
        return x
