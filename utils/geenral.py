
import time
import torch
import contextlib
import cv2
import numpy as np

class Timer(contextlib.ContextDecorator):
    def __init__(self, time=0.0):
        self.init_time(time)
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.init_time()
        self.start = self.get_time()
        return self

    def __exit__(self, type, value, traceback):
        self.time += self.get_time() - self.start

    def init_time(self, time=0.0):
        self.time = time
        
    def get_time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


def normalize_img(img, channel_first=True):
    if channel_first:
        img = np.swapaxes(img, 2, 1)
        img = np.swapaxes(img, 1, 0)
    img = img[np.newaxis, ...]/255.0
    img = img.astype(np.float32)
    return img



def save_weight_torch(model, path):
    model.eval()
    torch.save(model.state_dict(), path)

def load_weight_torch(model, path):
    model.load_state_dict(torch.load(path))
    return model

def save_onnx_from_torch(model_torch, path_onnx, inputs):
    torch.onnx.export(model_torch, inputs, path_onnx, input_names=["input_1"], output_names=["output_1"], export_params=True)


def onnx2trt():
    pass