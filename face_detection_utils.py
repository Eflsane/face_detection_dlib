import torch.nn as nn
import torchvision.transforms.functional as tf


class DepthwiseConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super(DepthwiseConv2d, self).__init__()

        self.depthwise = nn.Conv2d(input_channels, input_channels, kernel_size, groups=input_channels, bias=False,
                                   **kwargs)
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class EntryBlock(nn.Module):
    def __init__(self):
        super(EntryBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.conv3_residual = nn.Sequential(
            DepthwiseConv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            DepthwiseConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.conv3_direct = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )

        self.conv4_residual = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthwiseConv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            DepthwiseConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv4_direct = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        residual = self.conv3_residual(x)
        direct = self.conv3_direct(x)
        x = residual + direct

        residual = self.conv4_residual(x)
        direct = self.conv4_direct(x)
        x = residual + direct

        return x


class MiddleConvBlock(nn.Module):
    def __init__(self):
        super(MiddleConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthwiseConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256)
        )
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthwiseConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256)
        )
        self.conv3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthwiseConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        return x + residual


class MiddleBlock(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()

        self.block = nn.Sequential(*[MiddleConvBlock() for _ in range(num_blocks)])

    def forward(self, x):
        x = self.block(x)

        return x


class ExitBlock(nn.Module):
    def __init__(self):
        super(ExitBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.LeakyReLU(0.2),
            DepthwiseConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            DepthwiseConv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.direct = nn.Sequential(
            nn.Conv2d(256, 512, 1, stride=2),
            nn.BatchNorm2d(512)
        )

        self.conv = nn.Sequential(
            DepthwiseConv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            DepthwiseConv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )

        self.dropout = nn.Dropout(0.3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        direct = self.direct(x)
        residual = self.residual(x)
        x = direct + residual

        x = self.conv(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        return x


class XceptionNet(nn.Module):
    def __init__(self, num_middle_blocks=6):
        super(XceptionNet, self).__init__()

        self.entry_block = EntryBlock()
        self.middel_block = MiddleBlock(num_middle_blocks)
        self.exit_block = ExitBlock()

        self.fc = nn.Linear(1024, 136)

    def forward(self, x):
        x = self.entry_block(x)
        x = self.middel_block(x)
        x = self.exit_block(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def preprocess_image(image):
    image = tf.to_pil_image(image)
    image = tf.resize(image, (128, 128))
    image = tf.to_tensor(image)
    image = (image - image.min()) / (image.max() - image.min())
    image = (2 * image) - 1
    return image.unsqueeze(0)


class FaceLandmarksPart:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class FaceLandmarks:
    def __init__(self):
        self.parts: [FaceLandmarksPart] = []

    def append_part(self, x: int, y: int):
        self.parts.append(FaceLandmarksPart(x, y))

    def part(self, num: int) -> FaceLandmarksPart:
        part = self.parts[num]
        return part

    def total_parts(self) -> int:
        return self.parts.__len__()

