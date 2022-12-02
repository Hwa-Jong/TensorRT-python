import tensorflow as tf
from tf_models import VGG
import numpy as np

import tf2onnx
import onnx
import cv2

from utils import geenral
#from mmcv.tensorrt import onnx2trt, save_trt_engine



def train():
    batch_size = 4
    # make sample data 
    x_train = np.random.rand(100, 224,224,3).astype(np.float32)
    y_train = np.random.rand(100, 10)
    # data preprocessing
    x_train = x_train/255.0
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    model = VGG()
    model.build(input_shape=(batch_size, 224,224,3))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    epochs = 1
    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))
        # Iterate over the batches of the dataset.
        for iter, x_batch_train in enumerate(train_dataset):
            x = x_batch_train[0]
            y = x_batch_train[1]
            with tf.GradientTape() as tape:
                pred = model(x)
                # Compute reconstruction loss
                loss = loss_fn(y, pred)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    model.save_weights('vgg_tf.h5')
    #model.save('vgg_tf.h5')

    input_signature = (tf.TensorSpec((1,224,224,3), tf.float32, name="input_1"),)
    model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature=input_signature)
    onnx.save(model_proto, 'vgg_tf.onnx')
    
    '''## Create TensorRT engine
    max_workspace_size = 1 << 30
    opt_shape_dict = {
    'input': [[1,224,224,3],
              [1,224,224,3],
              [1,224,224,3]]
    }
    trt_engine = onnx2trt(
        model_proto,
        opt_shape_dict,
        fp16_mode=False,
        max_workspace_size=max_workspace_size)

    ## Save TensorRT engine
    save_trt_engine(trt_engine, 'tftest.trt')'''

    
    
    print('< test >')
    img = cv2.imread('sample_cat.jpg')
    img = geenral.normalize_img(img, channel_first=False)
    train_dataset = tf.data.Dataset.from_tensor_slices((img, None))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(1)
    for iter, x_batch_train in enumerate(train_dataset):
        x = x_batch_train[0]
        pred = model(x, training=False)

    pred = pred.numpy()
    print(pred)
    np.savetxt('vgg_tf_results.txt', pred)

    print('finish tensorflow train')



def main():
    train()

if __name__=='__main__':
    main()