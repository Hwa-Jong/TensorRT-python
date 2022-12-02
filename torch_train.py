from torch_models import VGG
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pytorch_model_summary as tsummary
import cv2

from utils import geenral


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dataset(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

def main():
    model = VGG()
    batch_size = 4
    print(tsummary.summary(model, torch.zeros((1,3,224,224)),batch_size=batch_size, show_input=False))
    epochs = 1
    trainloader, testloader, classes = dataset(batch_size)
    optimizer = optim.Adam(model.parameters(), lr=1E-4)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    #training
    for epoch in range(epochs):
        model.train()
        save_path="./vgg_torch"+ str(epoch+1) +".pt"
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if(loss.item() > 1000):
                print(loss.item())
                for param in model.parameters():
                    print(param.data)

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

            # for simple step...
            if i > 150:
                break

        # save model
        model.eval()
        geenral.save_weight_torch(model, 'vgg_torch.pt')
        geenral.save_onnx_from_torch(model, 'vgg_torch.onnx', inputs=inputs[0:1]) # to set batch_size = 1

        
        '''
        import onnx
        from mmcv.tensorrt import onnx2trt, save_trt_engine
        ## Create TensorRT engine
        max_workspace_size = 1 << 30
        opt_shape_dict = {
        'input': [list(inputs[0:1].shape),list(inputs[0:1].shape),list(inputs[0:1].shape)]
        }
        trt_engine = onnx2trt(
            onnx.load('torchtest.onnx'),
            opt_shape_dict,
            fp16_mode=False,
            max_workspace_size=max_workspace_size)

        ## Save TensorRT engine
        save_trt_engine(trt_engine, 'torchtest.trt')'''


    print('< test >')
    img = cv2.imread('sample_cat.jpg')
    img = geenral.normalize_img(img)
    img = torch.from_numpy(img)
    img = img.to(device)
    pred = model(img)
    pred = pred.detach().cpu().numpy()
    print(pred)
    np.savetxt('vgg_torch_results.txt', pred)

    print('Finished Training')

if __name__ == '__main__':
    main()