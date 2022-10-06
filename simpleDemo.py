import numpy
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm_notebook

from model import train
from model import test
from model import save
from model import load
from NetList  import Net_MCDO
'''
    Download the dataset , spilt it into traing/testing set 
'''
batch_size = 1000

train_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


log_freq = len(trainset)//batch_size
test_freq = 10
epoch_num = 200


losses = list()
net_scores = list()
test_scores = list()
is_train = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print (Net_MCDO.__name__)

network = Net_MCDO()
network.to(device)

network_test = Net_MCDO()
network_test.to(device)

network_test.load_state_dict(network.state_dict())

optimizer = optim.Adam(network.parameters(), lr=5e-4, weight_decay=0.0005, amsgrad=True)
schedular = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

for i in tqdm_notebook(range(epoch_num)):
    schedular.step()
    loss_avg = train(eopch=i, net=network, net_test=network_test, optimizer=optimizer,\
                     trainloader=trainloader, device=device, log_freq=log_freq)
    losses.append(loss_avg)
    if (i+1) % test_freq ==0:
        print ('Net_MCDO TEST')
        print ('Train net test')
        net_score = test(network, classes=classes, testloader=testloader, device=device, is_MCDO=True)
        net_scores.append(net_score)
        print ('Test net test')
        test_score = test(network_test, classes=classes, testloader=testloader, device=device, is_MCDO=True)
        test_scores.append(test_score)