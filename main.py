# pytorch进行CIFAR-10分类
# 依次按照下列顺序进行：
# 1.使用torchvision加载和归一化CIFAR-10训练集和测试集
# 2.定义一个卷积神经网络
# 3.定义损失函数和优化器
# 4.在训练集上训练网络
# 5.在测试集上测试网络

# 全局取消证书验证
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# 1.CIFAR-10数据加载和处理
# 导入torch
import torch
# 导入torchvision
import torchvision
# 进行数据预处理的模块
import torchvision.transforms as transforms

# 首先定义了一个变换transform，利用的是transforms模块中的Compose()
# 把多个变换组合在一起，可以看到这里组合了ToTensor和Normalize这两个变换
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 0.5：将这个三维的数据归一化成（-1，1）之内的范围
)

# 定义训练集trainset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# tranloader是按照batchsize大小来切分数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# 定义测试集testset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# 给定类别信息
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 展示图像
import matplotlib.pyplot as plt
import numpy as np


# 展示图像的函数
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 2.定义卷积神经网络
# 调用神经网络工具箱torch.nn、神经网络函数torch.nn.functional
import torch.nn as nn
import torch.nn.functional as F


# nn.Module是所有神经网络的基类
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 添加第一个卷积层，调用了nn里面的Conv2d()
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 同样是卷积层
        # 三个全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # relu是激活函数，引入非线性化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平，第一维度不变，后面的压缩
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义完类之后实例化一个net
net = Net()

# 3.定义损失函数和优化器
# 导入torch.potim优化器模块
import torch.optim as optim

# 用了神经网络工具箱nn中的交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# optim模块中的SGD梯度优化方式---随机梯度下降
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4.训练
if __name__ == '__main__':
    # epoch代表所有数据被训练的总轮数
    for epoch in range(2):
        # 定义一个变量方便我们对loss进行输出
        running_loss = 0.0
        # 这里遇到了第一步中出现的trainloader，代码传入数据
        # enumerate是python的内置函数，既获得索引也获得数据
        for i, data in enumerate(trainloader, 0):
            # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            inputs, labels = data
            # 把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度
            optimizer.zero_grad()
            # 把数据输进网络net
            outputs = net(inputs)
            # 计算损失值
            loss = criterion(outputs, labels)
            # loss进行反向传播
            loss.backward()
            # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮
            optimizer.step()
            # 打印loss
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # 5.测试
    # （1）随机读取4张图片，打印出相应的label信息
    # 创建一个python迭代器，读入的是我们第一步里面就已经加载好的testloader
    detailer = iter(testloader)
    # 返回一个batch_size的图片，根据第一步的设置，应该是4张
    images, labels = detailer.next()

    # 展示这四张图片
    imshow(torchvision.utils.make_grid(images))
    # python字符串格式化' '.join表示用空格来连接后面的字符串(python join()方法)
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    #（2）
    outputs = net(images)
    # output是一个多维向量，比如问题是10分类问题，我们输出就是一个10维度向量，
    # 每个元素代表该样本是某一类概率，10个元素里面最大的就是最大的概率，也就是指该样本最可能是的类别
    # torchmax作用就是找到最大概率对应的索引，就是预测的类别
    _, predicted = torch.max(outputs.data, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

    #（3）
    # 定义预测正确的图片数，初始化为0
    correct = 0
    # 总共参与测试的图片书，初始化为0
    total = 0
    # 循环每一个batch
    for data in testloader:
        images, labels = data
        # 输入网络进行测试
        outputs = net(images)
        # 我们选择概率最高的类作为预测
        _, predicted = torch.max(outputs.data, 1)
        # 更新测试图片的数量
        total += labels.size(0)
        # 更新预测正确的数量
        correct += (predicted == labels).sum()
    # 输出结果
    print(f'Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    # （4）准备计算每一类的预测
    # 字典生成器
    correct_pred = {classname: 0 for classname in classes}
    # 创建包含10个类别的字典
    total_pred = {classname: 0 for classname in classes}
    # 以一个batch为单位进行循环
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # 收集每个类别的正确预测
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    # 打印每一个类别的准确率
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
