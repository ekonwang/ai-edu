import torch.nn.functional
import torch
from sklearn import datasets
import torch.nn.functional as Fun
# 定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = Fun.relu(self.hidden(x))
        x = self.out(x)
        return x

def main():
    # net = MyNet()
    net = Net(n_feature=4, n_hidden=20, n_output=3)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    loss_func = torch.nn.CrossEntropyLoss()
    dataset = datasets.load_iris()
    input = torch.FloatTensor(dataset['data'])
    # print(input)
    label = torch.LongTensor(dataset['target'])
    # print(label)
    for i in range(1000):
        out = net(input)
        # print(out)
        loss = loss_func(out, label)
        # 输出与label对比
        optimizer.zero_grad()
        # 初始化
        loss.backward()
        optimizer.step()
    out = net(input)
    prediction = torch.max(out, 1)[1]
    pred_y = prediction.numpy()
    target_y = label.data.numpy()
    cnt=0
    for i in range(len(pred_y)):
        if pred_y[i]==target_y[i]:
            cnt+=1
    print(cnt/len(pred_y))#accuracy
main()
