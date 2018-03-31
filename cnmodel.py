import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from format import load_tf_data
from tqdm import *

# Hyper Parameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001
display_step = 1

train_loader, val_loader, test_loader = load_tf_data("ARID3A_K562_ARID3A_-sc-8821-_Stanford", batch_size=batch_size)
n_train = len(train_loader)
n_val = len(val_loader)
n_test = len(test_loader)

print n_train, n_val, n_test


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        def conv_block(in_channels, out_channels, conv_kernel, pool_kernel):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, conv_kernel, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(pool_kernel)
            )

        conv1 = 9
        maxpool1 = 2
        block1 = int((101 - conv1 + 1) / maxpool1)

        conv2 = 5
        maxpool2 = 2
        block2 = int((block1 - conv2 + 1) / maxpool2)

        conv3 = 3
        maxpool3 = block2 - conv3 + 1 # pool everything

        self.encoder = nn.Sequential(
            conv_block(1, 384, (conv1, 4), (maxpool1, 1)),
            conv_block(384, 384, (conv2, 1), (maxpool2, 1)),
            conv_block(384, 384, (conv3, 1), (maxpool3, 1)),
            Flatten(),
            nn.Linear(384, 2),
        )

    def forward(self, x):
        return self.encoder(x)

cnn = CNN()

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters())

# Train Model
for epoch in range(num_epochs):
    train_loss = 0.
    train_total = 0.
    train_correct = 0.
    for images, labels in tqdm(train_loader, total=n_train, desc="Epoch %d, Train" % (epoch+1)):
        images = Variable(images)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, Variable(labels))
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum()
        train_loss += loss.data[0]/n_train

    if epoch % display_step == 0:
        cnn.eval()
        correct = 0
        total = 0
        val_loss = 0.
        for images, labels in tqdm(val_loader, total=n_val, desc="Epoch %d, Valid" % (epoch+1)):
            images = Variable(images)

            outputs = cnn(images)

            loss = criterion(outputs, Variable(labels))

            val_loss += loss.data[0]/n_val
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print ('Epoch [%d] Train Loss: %.4f' % (epoch+1, train_loss))
        print ('Epoch [%d] Val Loss: %.4f' % (epoch+1, val_loss))
        print ('Epoch [%d] Train Acc: %.4f' % (epoch+1, float(train_correct) / train_total))
        print ('Epoch [%d] Val Acc: %.4f' % (epoch+1, float(correct) / total))
        cnn.train()

cnn.eval()

# Test Model

test_loss = 0.
test_total = 0.
test_correct = 0.

for images, labels in tqdm(test_loader, total=n_test):

    images = Variable(images)

    outputs = cnn(images)
    loss = criterion(outputs, Variable(labels))

    test_loss += loss.data[0]/n_test
    _, predicted = torch.max(outputs.data, 1)
    test_total += labels.size(0)
    test_correct += (predicted == labels).sum()

print ('Test Loss: %.4f' % (test_loss))
print ('Test Acc: %.4f' % (float(test_correct) / test_total))

# Save Trained Model
torch.save(cnn.state_dict(), 'cnn.pk1')
