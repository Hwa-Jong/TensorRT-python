import torch
import tensorflow as tf

print('< torch >')
print(torch.__version__)
print(torch.cuda.is_available())
print()

print('< tensorflow >')
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
print()


