import torch
import cv2
import numpy as np
import onnxruntime
import onnx


import tensorflow as tf
from tensorflow.keras import Model, layers, Input

from torch_models import VGG19
from tf_models import VGG_tf
from utils.geenral import Timer

from mmcv.tensorrt import TRTWrapper

iter_num = 1000
def main():
    torch_time = {'load_data' : Timer(), 'load_model' : Timer(), 'inference' : Timer(), 'ToGPU' : Timer(), 'total':Timer() }
    torch_onnx_time = {'load_data' : Timer(), 'load_model' : Timer(), 'inference' : Timer(), 'ToGPU' : Timer(), 'total':Timer() }
    torch_trt_time = {'load_data' : Timer(), 'load_model' : Timer(), 'inference' : Timer(), 'ToGPU' : Timer(), 'total':Timer() }
    tf_time = {'load_data' : Timer(), 'load_model' : Timer(), 'inference' : Timer(), 'ToGPU' : Timer(), 'total':Timer() }
    tf_onnx_time = {'load_data' : Timer(), 'load_model' : Timer(), 'inference' : Timer(), 'ToGPU' : Timer(), 'total':Timer() }
    tf_trt_time = {'load_data' : Timer(), 'load_model' : Timer(), 'inference' : Timer(), 'ToGPU' : Timer(), 'total':Timer() }
       
    torch_show = True
    torch_onnx_show = True
    torch_trt_show = True
    tf_show = False
    tf_onnx_show = False
    tf_trt_show = False

    if torch_show:
        torch_time, pred = torch_test(torch_time)
        print('< torch result >')
        show_tmr(torch_time)
        print(pred)
        print('=================================================================')
    
    if torch_onnx_show:
        torch_onnx_time, pred = torch_onnx_test(torch_onnx_time)
        print('< torch onnx result >')
        show_tmr(torch_onnx_time)
        print(pred)
        print('=================================================================')
        
    if torch_trt_show:
        torch_trt_time, pred = torch_trt_test(torch_trt_time)
        print('< torch trt result >')
        show_tmr(torch_trt_time)
        print(pred)
        print('=================================================================')
        
    if tf_show:
        tf_time, pred = tf_test(tf_time)
        print('< tf result >')
        show_tmr(tf_time)
        print(pred)
        print('=================================================================')
        
    if tf_onnx_show:
        tf_onnx_time, pred = tf_onnx_test(tf_onnx_time)
        print('< tf onnx result >')
        show_tmr(tf_onnx_time)
        print(pred)
        print('=================================================================')

    if tf_trt_show:
        tf_trt_time, pred = tf_trt_test(tf_trt_time)
        print('< tf trt result >')
        show_tmr(tf_trt_time)
        print(pred)
        print('=================================================================')

def show_tmr(tmr):
    print('load data time : ', tmr['load_data'].time) 
    print('load model time : ', tmr['load_model'].time)
    print('ToGPU time : ', tmr['ToGPU'].time) 
    print(str(iter_num) + ' inference time : ', tmr['inference'].time) 
    print('total time : ', tmr['total'].time)


def torch_test(tmr):
    device = 'cuda:0'
    with tmr['total']:
        with tmr['load_model']:
            model = VGG19()
            #model.load_state_dict(torch.load('torch_vgg_1.pth'))
            model.load_state_dict(torch.load('torchtest.pth'))
            model.eval()

        with tmr['load_data']:
            img = cv2.imread('test224.png')
            img = np.swapaxes(img, 2, 1)
            img = np.swapaxes(img, 1, 0)
            img = img[np.newaxis, ...]/255.0
            img = torch.from_numpy(img.astype(np.float32))

        with tmr['ToGPU']:
            img = img.to(device)
            model.to(device)

        with tmr['inference']:
            for i in range(iter_num):
                pred = model(img)
        

    return tmr, pred.detach().cpu().numpy()


def torch_onnx_test(tmr):
    with tmr['total']:
        with tmr['load_model']:
            #session = onnxruntime.InferenceSession("torch_VGG224_fix.onnx")
            session = onnxruntime.InferenceSession("torchtest.onnx")

        with tmr['load_data']:
            img = cv2.imread('test224.png')
            img = np.swapaxes(img, 2, 1)
            img = np.swapaxes(img, 1, 0)
            img = img[np.newaxis, ...]/255.0
            img = img.astype(np.float32)

        with tmr['ToGPU']:
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

        with tmr['inference']:
            for i in range(iter_num):
                pred = session.run([output_name], {input_name : img})[0]
    
    return tmr, pred

def torch_trt_test(tmr):
    device = 'cuda:0'
    with tmr['total']:
        with tmr['load_data']:
            img = cv2.imread('test224.png')
            img = np.swapaxes(img, 2, 1)
            img = np.swapaxes(img, 1, 0)
            img = img[np.newaxis, ...]/255.0
            img = img.astype(np.float32)
            img = torch.from_numpy(img)

        with tmr['load_model']:
            #trt_model = TRTWrapper('torch_VGG224_fix.trt', ['input'], ['output'])
            trt_model = TRTWrapper('torchtest.trt', ['input'], ['output'])

        with tmr['ToGPU']:
            img = img.to(device)
            trt_model = trt_model.to(device)

        with tmr['inference']:
            with torch.no_grad():
                for i in range(iter_num):
                    trt_outputs = trt_model({'input': img})
                    pred = trt_outputs['output']
        
    return tmr, pred


def tf_test(tmr):
    device = tf.config.experimental.list_physical_devices('GPU')
    with tmr['total']:
        with tmr['load_model']:
            '''model = VGG_tf()
            inputs = Input(shape=(224,224,3))
            model = Model(inputs=inputs, outputs=model(inputs))
            model.load_weights('tf_vgg_save_dir/tf_vgg')'''
            #model = tf.keras.models.load_model('tf_vgg_save_dir.h5', custom_objects={'VGG_tf':VGG_tf})
            model = tf.keras.models.load_model('tftest.h5', custom_objects={'VGG_tf':VGG_tf})

        with tmr['load_data']:
            img = cv2.imread('test224.png')
            img = img/255.0
            img = img.astype(np.float32)

        with tmr['ToGPU']:
            data = tf.data.Dataset.from_tensor_slices((img[np.newaxis, ...], None))
            data = data.shuffle(buffer_size=1024).batch(1)
            for iter, x_batch in enumerate(data):
                img = x_batch[0]

        with tmr['inference']:
            for i in range(iter_num):
                pred = model(img, training=False)
    
    return tmr, pred

def tf_onnx_test(tmr):
    with tmr['total']:
        with tmr['load_model']:
            #session = onnxruntime.InferenceSession("tf_VGG224_fix.onnx")
            session = onnxruntime.InferenceSession("tftest.onnx")

        with tmr['load_data']:
            img = cv2.imread('test224.png')
            img = img[np.newaxis, ...]/255.0
            img = img.astype(np.float32)

        with tmr['ToGPU']:
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

        with tmr['inference']:
            for i in range(iter_num):
                pred = session.run([output_name], {input_name : img})[0]

    return tmr, pred

def tf_trt_test(tmr):
    device = 'cuda:0'
    with tmr['total']:
        with tmr['load_data']:
            img = cv2.imread('test224.png')
            img = img[np.newaxis, ...]/255.0
            img = img.astype(np.float32)
            img = torch.from_numpy(img)

        with tmr['load_model']:
            trt_model = TRTWrapper('tf_VGG224_fix.trt', ['input'], ['vgg_tf'])

        with tmr['ToGPU']:
            img = img.to(device)
            trt_model = trt_model.to(device)

        with tmr['inference']:
            with torch.no_grad():
                for i in range(iter_num):
                    trt_outputs = trt_model({'input': img})
                    pred = trt_outputs['vgg_tf']
        
    return tmr, pred


if __name__ =='__main__':
    main()