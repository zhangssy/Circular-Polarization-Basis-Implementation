import os
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from UNet_11 import *


plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  

batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path of datasets
train_txt_path = os.path.join("train_txt_path", "Train.txt")
test_txt_path = os.path.join("test_txt_path", "Test.txt")
# learning accuracy curve
save_path = "accuracy_curve_path.png"


class MyDataset(Dataset):
    def __init__(self, txt_path):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)


# 构建数据集和数据加载器
train_data = MyDataset(txt_path=train_txt_path)
test_data = MyDataset(txt_path=test_txt_path)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def calc_loss(outputs, labels, device):
    outputs = outputs.to(device)
    labels = labels.to(device)
    criterion = nn.CrossEntropyLoss()
    loss1 = criterion(outputs, labels)
    return loss1.mean()


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    count = 0

    for i, (inputs_1, labels) in enumerate(train_loader):
      
        traininputs0 = torch.rand(1, 9, 11, 11).to(device)
        traininputs = traininputs0

    
        for j in range(len(inputs_1)):
            inputs_2 = torch.load(inputs_1[j]).cuda()
            traininputs = torch.cat((traininputs, inputs_2), 0).cuda()

        traininputs4 = del_tensor_ele(traininputs, 0).cuda()
        del traininputs

        
        optimizer.zero_grad()
        outputs = model(traininputs4).cuda()
        loss = calc_loss(outputs, labels, device).cuda()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 计算训练精度
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            test_labels = labels.numpy()
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            test_labels = np.concatenate((test_labels, labels.numpy()))

  
    correct = np.sum(test_labels == y_pred_test)
    acc = round(correct / len(y_pred_test) * 100, 2)
    print(f"{epoch + 1} - train_acc: {acc}%")
    return acc


def test(model, device, test_loader):
    model.eval()
    count = 0

    with torch.no_grad():  
        for inputs_1, labels in test_loader:
          
            testinputs0 = torch.rand(1, 9, 11, 11).to(device)
            testinputs = testinputs0

           
            for j in range(len(inputs_1)):
                inputs_2 = torch.load(inputs_1[j]).cuda()
                testinputs = torch.cat((testinputs, inputs_2), 0).cuda()

            testinputs4 = del_tensor_ele(testinputs, 0).cuda()
            del testinputs

         
            outputs = model(testinputs4).cuda()
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)

           
            if count == 0:
                y_pred_test = outputs
                test_labels = labels.numpy()
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                test_labels = np.concatenate((test_labels, labels.numpy()))

    
    correct = np.sum(test_labels == y_pred_test)
    acc = round(correct / len(y_pred_test) * 100, 2)
    print(f"{epoch + 1} - test_acc: {acc}%")
    return acc



num_classes = 7
model = UNet(9, 7).to(device)
momentum = 0.9
betas = (0.9, 0.999)
num_epochs = 200
best_acc = 0.0
lr = 0.0001


train_accuracies = []
test_accuracies = []


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer, T_max=num_epochs
)


for epoch in range(num_epochs):
    if epoch % 50 == 0 and epoch != 0:
        lr *= 0.1
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=num_epochs - epoch
        )

    train_acc = train(model, device, train_loader, optimizer, epoch)
    train_accuracies.append(train_acc)

    optimizer.step()
    scheduler.step()


    test_acc = test(model, device, test_loader)
    test_accuracies.append(test_acc)


    if test_acc >= best_acc:
        best_acc = test_acc
        torch.save(
            model.state_dict(),
            f'save model path'
        )

print("Finished!")


plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training', marker='o', markersize=3)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Testing', marker='s', markersize=3)
plt.xlabel('Epochs')
plt.ylabel('OA (%)')
plt.title('Learning curve on circular polarization basis')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"finished: {save_path}")


plt.show()