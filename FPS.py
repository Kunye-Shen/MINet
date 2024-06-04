import numpy as np
import torch
import time
from tqdm import tqdm

def computeTime(model, device='cuda'):
    inputs = torch.randn(300, 3, 368, 368)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    time_spent = []
    for idx in tqdm(range(100)):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 20:
            time_spent.append(time.time() - start_time)
    print('Average speed: {:.4f} fps'.format(300 / np.mean(time_spent)))


torch.backends.cudnn.benchmark = True

from model import MINet
model = MINet()

computeTime(model)
