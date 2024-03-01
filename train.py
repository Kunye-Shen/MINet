import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import glob

from data_loader import *

from model import MINet

import pytorch_ssim

# ------- processes the training dataset -------
dataset_dir = "./Dataset/SD-saliency-900/"
image_train_dir = dataset_dir + "Img_train/"
label_train_dir = dataset_dir + "GT_train/"

image_ext = '.bmp'
label_ext = '.png'

model_dir = "./model_save/"

epoch_num = 1803
batch_size_train = 32

image_train_name_list = glob.glob(image_train_dir + '*' + image_ext)

label_train_name_list = []
for image_path in image_train_name_list:
	image_name = image_path.split("/")[-1]

	t = image_name.split(".")
	t = t[0:-1]
	imidx = t[0]
	for i in range(1,len(t)):
		imidx = imidx + "." + t[i]

	label_train_name_list.append(label_train_dir + imidx + label_ext)

print("---")
print("train images: ", len(image_train_name_list))
print("train labels: ", len(label_train_name_list))
print("---")

train_num = len(image_train_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=image_train_name_list,
    lbl_name_list=label_train_name_list,
    transform=transforms.Compose([
        Rescale(368),
        RandomCrop(336),
        ToTensor(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)

# ------- define loss function -------
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)

def hybrid_loss(pred,target):

    bce_out = bce_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)

    loss = bce_out + ssim_out

    return loss

def multi_loss_function(d1, d2, d3, d4, d5, labels_v):

    loss1 = hybrid_loss(d1,labels_v)
    loss2 = hybrid_loss(d2,labels_v)
    loss3 = hybrid_loss(d3,labels_v)
    loss4 = hybrid_loss(d4,labels_v)
    loss5 = hybrid_loss(d5,labels_v)

    loss = loss1 + loss2 + loss3 + loss4 + loss5

    return loss

# ------- define model --------
# define the net
net = MINet()
if torch.cuda.is_available():
    net.cuda()

# ------- define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=4e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- training process --------
print("---start training...")
ite_num = 0

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        optimizer.zero_grad()

        d1, d2, d3, d4, d5 = net(inputs_v)
        loss = multi_loss_function(d1, d2, d3, d4, d5, labels_v)

        loss.backward()
        optimizer.step()

        if ite_num % 1000 == 0:  # save model every 1000 iterations
            torch.save(net.state_dict(), model_dir + "epoch_%d_items_%d.pth" % (epoch, ite_num))

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f" % 
        (epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, loss.item()))

        del d1, d2, d3, d4, d5, loss

print('-------------Congratulations! Training Done!-------------')