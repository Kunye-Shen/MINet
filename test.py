from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
import glob
from tqdm import tqdm

from data_loader import *

from model import MINet


def normPRED(x):
    MAX = torch.max(x)
    MIN = torch.min(x)

    out = (x - MIN) / (MAX - MIN)

    return out


def save_output(image_dir, image_name, pred, save_dir):
    predict = pred
    predict = predict.squeeze()
    predict = predict.cpu().data.numpy()
    predict = Image.fromarray(predict * 255).convert('RGB')

    image = io.imread(image_dir + image_name + '.bmp')
    predict = predict.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    predict.save(save_dir + image_name + '.png')


if __name__ == '__main__':
    # --------- Define the address and image format ---------
    image_dir = "./Dataset/SD-saliency-900/Img_test/"
    prediction_dir = "./results/"
    model_dir = "./model_save/MINet.pth"

    img_name_list = glob.glob(image_dir + '*.bmp')

    # --------- Load the data ---------
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[], transform=transforms.Compose([Rescale(368), ToTensor(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=64, shuffle=False, num_workers=4)

    # --------- Define the model ---------
    print("...load MINet...")
    net = MINet()
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- Generate prediction images ---------
    for i_test, data_test in tqdm(enumerate(test_salobj_dataloader)):
        inputs_test, name_list = data_test['image'], data_test['name']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5 = net(inputs_test)

        # normalization
        for i in range(d1.shape[0]):
            pred = d1[i, 0, :, :]
            pred = normPRED(pred)
            save_output(image_dir, name_list[i], pred, prediction_dir)

        del d1, d2, d3, d4, d5
