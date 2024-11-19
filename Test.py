import torch.nn as nn
from LAuReL import LAuReLBlock
from TrainUtil import train_and_evaluate
# A
# from LAuReL_PABlock import LAuReL_PABlock
# from LAuReL_LiteBlock import LAuReL_LiteBlock

# Simple CNN with LAuReL Blocks
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.laurel_block = LAuReLBlock(in_channels=32, out_channels=32, )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.laurel_block(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    # Train and evaluate with LAuReL
    model_lr = SmallCNN()
    print("Training with LAuReL Block")
    train_and_evaluate(model_lr)

    # Train and evaluate with LAuReL_LR
    # model_pa = SmallCNN() # with LAuReL_LRBlock changed implemenation
    # train_and_evaluate(model_pa)

    # Train and evaluate with LAuReL_PA
    # model_pa = SmallCNN() # with LAuReL_PABlock changed implemenation
    # train_and_evaluate(model_pa)
