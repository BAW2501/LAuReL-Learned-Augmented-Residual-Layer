import torch.nn as nn
from LAuReL import LAuReLBlock
from LAuReL_PA import LAuReL_PABlock
from LAuRel_LR import LAuReL_LRBlock
from ResNet import ResBlock
from TrainUtil import train_and_evaluate

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
    
class SmallCNNLR(SmallCNN):
    def __init__(self):
        super().__init__()
        self.laurel_block = LAuReL_LRBlock(in_channels=32, out_channels=32)
        
class SmallCNNPA(SmallCNN):
    def __init__(self):
        super().__init__()
        self.laurel_block = LAuReL_PABlock(in_channels=32, out_channels=32)
    def reset_activations(self):
        """Reset the activations in all LAuReL-PA blocks"""
        self.laurel_block.reset_activations()
        
class SmallCNNResNet(SmallCNN):
    def __init__(self):
        super().__init__()
        self.laurel_block = ResBlock(in_channels=32, out_channels=32)
       
    
if __name__ == "__main__":
    # Train and evaluate with LAuReL
    model_lr = SmallCNN()
    print("Training with LAuReL Block")
    train_and_evaluate(model_lr)

    # Train and evaluate with LAuReL_LR
    model_pa = SmallCNNLR() # with LAuReL_LRBlock changed implemenation
    train_and_evaluate(model_pa)

    # # Train and evaluate with LAuReL_PA
    # model_pa = SmallCNNPA() # with LAuReL_PABlock changed implemenation
    # train_and_evaluate(model_pa)
    
    # # Train and evaluate with ResNet
    model_pa = SmallCNNResNet() # with ResBlock changed implemenation
    train_and_evaluate(model_pa)
