import numpy as np
import torch
import torch.nn as nn
from NetList import update_target

CE = nn.CrossEntropyLoss()

def train(epoch, net, net_test, optimizer, trainloader, device, log_freq=10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward+ backward+ optimize
        outputs = net(inputs)
        loss = CE(outputs, labels)
        loss.backward()
        optimizer.step()
        update_target(net_test, net, 0.001)

        # print statistics
        running_loss += loss.item()
        if (i+1) % log_freq == 0:   # print every 2000 mini-batches
            print ('[Epoch : %d, Iter: %5d] loss: %.3f' %(epoch +1 , i+1, running_loss/log_freq))

    return running_loss / log_freq


def test(net, classes, testloader, device, is_MCDO = False):
    print ('Start testng')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output = 0
            if is_MCDO:
                for i in range(10):
                    output += net(inputs)/10.
                output = torch.log(output)
            else:
                output = net(inputs)
            _, predicted = torch.max(output, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print ('Accuracy of %5s : %.2f %%' %(
            classes[i], 100 * class_correct[i] / class_total[i]
        ))
    test_score = np.mean([100* class_correct[i] / class_total[i] for i in range(10)])
    print (test_score)
    return test_score

#  SAVE
def save(name, net, net_test, device):
    net_path = './model/'+name+' .pkl'
    net_test_path = './model/'+name+'_test.pkl'

    net = net.cpu()
    torch.save(net.state_dict(), net_path)

    net_test = net_test.cpu()
    torch.save(net.state_dict(), net_test_path)
    # Place it to GPU back
    net.to(device)
    net_test.to(device)
    return net, net_test

def load(name, net, net_test, device):
    net_path = './model/'+name+'.pkl'
    net_test_path = './model'+name+'_test.pkl'
    # LOAD
    net.load_state_dict(torch.load(net_path))
    net_test.load_state_dict(torch.load(net_test_path))
    # Place it to GPU
    net.to(device)
    net_test.to(device)
    return net, net_test



