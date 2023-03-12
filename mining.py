import torch
import torchvision
import torchvision.transforms as transforms
import random
import os
import numpy as np

#fix the random seed of pytorch
def seed_torch(seed=1029):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed) # ban the randomness of hash
  np.random.seed(seed)
  torch.manual_seed(seed)
  #print(torch.cuda.is_available())
  if torch.cuda.is_available()==True :
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True
      
device = torch.device("cuda")
seed_torch()

'''
#MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
'''

#CIFAR
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainset1=list(enumerate(trainset,0))
trainloader = torch.utils.data.DataLoader(trainset1, batch_size=4,
                                          shuffle=True, num_workers=0)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

import torch.nn as nn
import torch.nn.functional as F
from model import ResNet18
#from model import MobileNetV2
import torch.optim as optim
from pol import *

net = ResNet18().to(device)
#net = MobileNetV2().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

#Directory where the mined blocks are stored
DIR = './PoLchain/'
if os.path.isdir(DIR):
    pass
else:
    os.makedirs(DIR)
def save_block(number,requirements,parameters_proof,indexs_proof):
    PATH=DIR+'block'+str(number)
    torch.save({
      'timestamp':time.time(),
      'indexs_proof': indexs_proof,
      'net_state_dict': parameters_proof,
      'number': number,
      'requirements':requirements
    },PATH)

def load_block(number):
    PATH=DIR+'block'+str(number)
    checkpoint = torch.load(PATH)
    return checkpoint
from torch.utils.data.dataloader import default_collate

#To verify the validation of a block, indicated by block number
def block_verify(number):
    checkpoint=load_block(number)
    global net
    net.load_state_dict(checkpoint['net_state_dict'])
    requirements=checkpoint['requirements']
    indexs_proof=checkpoint['indexs_proof']
    before=save_parameters(net.parameters,requirements)
    batch = default_collate([trainset[i] for i in indexs_proof])
    ##data0=[trainset[index][0] for index in indexs_proof]
    ##data1=[trainset[index][1] for index in indexs_proof]
    ##data=[data0,data1]
    ##print(batch)
    train_data=batch
    net=training(net,train_data)
    now=save_parameters(net.parameters,requirements)
    print('weights before:',before)
    print('weights now:',now)
    print('Requirements:',requirements)
    if not check_require(before,now,requirements):
        print("Requirement not satisfied")
        return False
    print("Requirement of Directed Guiding Gradient satisfied")
    if block_accuracy(number)<block_accuracy(number-1):
        print("Accuracy not satisfied")
        return False
    print("Requirement of Accuracy satisfied")
    print("Verified a valid block: Block %d"%number)
    return True

def block_accuracy(number):
    checkpoint=load_block(number)
    global net
    net.load_state_dict(checkpoint['net_state_dict'])    
    correct = 0
    total = 0
    test_number=1000 #control the number of test cases
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if total>test_number:
            break

    print('Accuracy: %f %%' % (100.0 * correct / total))
    return 100.0 * correct / total

def test_accuracy():
    global net
    correct = 0
    total = 0
    test_number=500
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if total>test_number:
            break
    return 100.0 * correct / total

#To start mining from the last block
def continue_mining(number):
    checkpoint=load_block(number)
    global net
    net.load_state_dict(checkpoint['net_state_dict'])
    requirements=checkpoint['requirements']
    difficulty=5
    for i in range(number+1,2000):
        hash_prev=get_parameter_hash(save_parameters(net.parameters,requirements))
        requirements=get_requirements(get_net_size(net.parameters()),difficulty,hash_prev)
        show_requirements(requirements)
        ##print(save_parameters(net.parameters,requirements))
        [parameters_proof,indexs_proof]=mining(net,trainset,requirements)
        print("Mined a new block whose mini-batch is:",indexs_proof)
        save_block(i,requirements,parameters_proof,indexs_proof)

#mining until find a valid proof
#time_training=0
#time_mining=0
#round_times=0
def mining(net,dataset,requirements):
    #global time_training
    #global time_mining
    #global round_times
    #global running_time
    while True:
        for i, data in enumerate(trainloader, 0):
            #stop_running(computing_power_percentage,running_time)
            #running_time=time.time()
            before=save_parameters(net.parameters,requirements)
            #time1=time.time()
            before_para=net.state_dict()
            torch.save(before_para,'temp') #to prevent torch.save lazy, can be removed if not invloving verify
            indexs,train_data=data
            #time2=time.time()
            net=training(net,train_data)
            #time3=time.time()
            now=save_parameters(net.parameters,requirements)
            #time4=time.time()
            #round_times+=1
            #time_training+=(time3-time2)
            #time_mining+=(time4-time3)+(time2-time1)
            if check_require(before,now,requirements):
                print('key weights before:',before)
                print('key weights now:',now)
                before_para=torch.load('temp') #to prevent torch.save lazy, can be removed if not invloving verify
                return [before_para,indexs]

##one step training
def training(net,data):
    inputs,labels=data
    
    inputs, labels = inputs.to(device), labels.to(device)
    
    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    return net

#Simulate the behavior of malicious miners with fixed requirements
def spoofing(net,dataset,requirements):
    #global time_training
    #global time_mining
    #global round_times
    accu1=test_accuracy()
    print(accu1)
    for i, data in enumerate(trainloader, 0):
        indexs,train_data=data
        while True:
            before=save_parameters(net.parameters,requirements)
            #time1=time.time()
            before_para=net.state_dict()
            #torch.save(before_para,'temp') #to prevent torch.save lazy 
            before_para['conv1.0.weight'][0][0][0][0]+=random.random()-0.5
            before_para['conv1.0.weight'][0][0][0][1]+=random.random()-0.5
            before_para['conv1.0.weight'][0][0][0][2]+=random.random()-0.5
            before_para['conv1.0.weight'][0][0][1][0]+=random.random()-0.5
            before_para['conv1.0.weight'][0][0][1][1]+=random.random()-0.5
            #time2=time.time()
            #time3=time.time()
            net.load_state_dict(before_para)
            now=save_parameters(net.parameters,requirements)
            #time4=time.time()
            #round_times+=1
            #time_training+=(time3-time2)
            #time_mining+=(time4-time3)+(time2-time1)
            if test_accuracy()<accu1:
                print(test_accuracy())
                continue
            if check_require(before,now,requirements):
                print('key weights before:',before)
                print('key weights now:',now)
                before_para=torch.load('temp') #to prevent torch.save lazy 
                return [before_para,indexs]
        break
        
#use running time to express computing power
#computing_power_percentage=1
#running_time=time.time()  
difficulty=8 #initial difficulty
h0=sha256("1234".encode("utf-8")).hexdigest() #hash of genesis block
requirements=get_requirements(get_net_size(net.parameters()),difficulty,h0) #initial requirements
#time_training=0
#time_mining=0
#round_times=0
for i in range(5):
    hash_prev=get_parameter_hash(save_parameters(net.parameters,requirements))
    requirements=get_requirements(get_net_size(net.parameters()),difficulty,hash_prev)
    show_requirements(requirements)
    #print(save_parameters(net.parameters,requirements))
    [parameters_proof,indexs_proof]=mining(net,trainset,requirements)
    print("Mined a new block whose mini-batch is:",indexs_proof)
    save_block(i,requirements,parameters_proof,indexs_proof)
    #print(time_training/round_times)
    #print(time_mining/round_times)
    
    
'''
#Verify a block    
block_verify(2)
#Simulate spoofing
difficulty=5 #initial difficulty
requirements=[0,2,4,6,8]
time_list=[]
for i in range(10):
    show_requirements(requirements)
    time0=time.time()
    [parameters_proof,indexs_proof]=spoofing(net,trainset,requirements)
    time_list.append(time.time()-time0)
    print(time.time()-time0)
    time0=time.time()
    print("Mined a new block whose mini-batch is:",indexs_proof)
#To calculate block interval
time_before=0
time_list=[]
for i in range(2000):
    if i%2==0:
        checkpoint=load_block(i)
        time_now=checkpoint['timestamp']
        time_list.append(time_now-time_before)
        time_before=time_now
#To calculate block accuracy
accuracy_list=[]
for i in range(2000):
     if i % 10==0:
        acc=block_accuracy(i)
        accuracy_list.append(acc)
print(accuracy_list)

'''  