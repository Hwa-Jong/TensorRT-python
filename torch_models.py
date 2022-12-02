import torch
import torch.nn as nn
import torch.nn.functional as F
#import pytorch_model_summary as tsummary

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn0_0 = nn.BatchNorm2d(64)
        self.conv0_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn0_1 = nn.BatchNorm2d(64)
        self.conv1_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1_0 = nn.BatchNorm2d(128)
        self.conv1_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(128)
        self.conv2_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2_0 = nn.BatchNorm2d(256)
        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(256)
        self.conv3_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn3_0 = nn.BatchNorm2d(512)
        self.conv3_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(512)
        self.conv3_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(512)
        self.conv4_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4_0 = nn.BatchNorm2d(512)
        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.bn0_0(self.conv0_0(x)))
        x = F.relu(self.bn0_1(self.conv0_1(x)))
        x = nn.MaxPool2d(2,2)(x)
        x = F.relu(self.bn1_0(self.conv1_0(x)))
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = nn.MaxPool2d(2,2)(x)
        x = F.relu(self.bn2_0(self.conv2_0(x)))
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = nn.MaxPool2d(2,2)(x)
        x = F.relu(self.bn3_0(self.conv3_0(x)))
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = nn.MaxPool2d(2,2)(x)
        x = F.relu(self.bn4_0(self.conv4_0(x)))
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = nn.MaxPool2d(8,8)(x)
        x = nn.Flatten()(x)
        x = self.fc(x)
        x = nn.functional.softmax(x, dim=1)
        return x

def main():
    model = VGG()
    print('finish')

if __name__=='__main__':
    main()