from __future__ import print_function, division

import io
import sys

import PIL.Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# import multiclass_f1_score
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryPrecision
import random
import argparse
from torchsummary import summary
from torch.autograd import Variable
import json
from flask import Flask
from flask import request, jsonify
import logging
from datetime import datetime
#log_file_name="log/app/" + datetime.now().strftime("%Y%m%d_%H%M%S")+".log"
log_file_name="log/app/" + datetime.now().strftime("%Y%m%d")+".log"
logging.basicConfig(filename=log_file_name,level=logging.INFO)
import threading
#from multiprocessing import Pool, Value,Lock


app = Flask(__name__)





shared_resource_lock=threading.Lock()


def tensorToNpImage(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

class SingleImage:
    def __init__(self, path, image, trans_image, size):
        self.filename = path
        self.image = image
        self.trans_image = trans_image

def start_detect(device, class_names, model, dataloaders, dataset_size, show_image, single_image=None):
    model.eval()
    images_so_far = 0
    images = []
    imageInfo = []
    count = 0
    detect_folder = generate_new_detectfolder()

    with torch.no_grad():
        if (single_image != None):  # classify single image by assinged file path
            image = single_image.trans_image
            image = Variable(image, requires_grad=True)
            image = image.unsqueeze(0)
            outputs = model(image.cuda())
            _, preds = torch.max(outputs, 1)
            images.append(single_image.trans_image)
            filename = single_image.filename.split("/")[-1]
            imageInfo.append({'info': f'{filename}\ndetect: {class_names[preds[0]]}',
                              'rslt': class_names[preds[0]],
                              'filename': filename
                              })
        else:  # classify batch images by assigned folder path
            for i, (inputs, labels) in enumerate(dataloaders):
                inputs = inputs.to(device)
                # labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                # display each image step-wisely
                for j in range(inputs.size()[0]):
                    # count = count + 1
                    images_so_far += 1
                    if show_image:
                        images.append(inputs.cpu().data[j])
                    filename = dataloaders.dataset.imgs[images_so_far - 1][0].split('/')[-1]
                    imageInfo.append({'info': f'{filename}\ndetect: {class_names[preds[j]]}',
                                      'rslt': class_names[preds[j]],
                                      'filename': filename
                                      })

    write_result(imageInfo, detect_folder)

    axs = []
    print("show image:")
    print(show_image)
    if show_image:
        for i in range(0, len(images)):
            ax = plt.figure()
            img = tensorToNpImage(images[i])
            plt.title(imageInfo[i]['info'], size=12, color='black')
            if imageInfo[i]['rslt'] == 0:
                plt.title(imageInfo[i]['info'], size=12, color='grey', fontweight="bold")
            else:
                plt.title(imageInfo[i]['info'], size=12, color='black')
            plt.imshow(img)
            plt.savefig(detect_folder + "/" + imageInfo[i]['filename'] + ".jpg")
            plt.show()
            plt.close()


def write_result(image_info, dir_path):
    json_object = json.dumps(image_info)
    newdir = dir_path
    with open(dir_path + "/rslt.json", "w") as outfile:
        outfile.write((json_object))

    dct = {x['filename']: x for x in image_info}

    json_object = json.dumps(dct)
    with open(dir_path + "/dct_rslt.json", "w") as outfile:
        outfile.write((json_object))



'''
if __name__ == '__main__':
    defaultModelPath = 'models/pytorch/resnet50_adam_20epochs_weights.h5'
    defaultDataPath = 'data/ToDetect'
    # defaultDataPath = 'data/ToDetect/images/136314.jpg'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', required=False, type=int, default=2, help='batch size')
    parser.add_argument('--weight-path', required=False, type=str, default=defaultModelPath,
                        help='path of model weight')
    parser.add_argument('--data-path', required=False, type=str, default=defaultDataPath,
                        help='path of detection data folder')
    parser.add_argument('--device', required=False, type=str, default="gpu", help='gpu or cpu')
    parser.add_argument('--show-image', required=False, type=bool, default=False,
                        help='True or False (sample 12 images to display')
    parser.add_argument('--class-lable', required=False, type=str, default="Blurry/Clear",
                        help='input fomat: label1,label2,...,labelN (default:Blurry,Clear)')
    parser.add_argument('--imgsize', required=False, type=int, default=448,
                        help='size to resize and cropsize (size*size, default 448*448)')
    args = parser.parse_args()
    detect(selected_device=args.device,
           batch_size=args.batch_size,
           weight_path=args.weight_path,
           data_dir=args.data_path,
           show_image=args.show_image,
           class_name_str=args.class_lable,
           image_size=args.imgsize
           )

'''

def init_model(selected_device='gpu', weight_path='models/pytorch/resnet50_adam_20epochs_weights.h5',
               data_dir='data/ToDetect', batch_size=1, show_image=False, class_name_str="Blurry/Clear", image_size=448):
    # data_dir="/home/bryanni/DlBlurDetection/BlurDetectEnv/data/ToDetect/20230412_FaceCamImage"
    # device setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if selected_device == 'cpu':
        device = torch.device("cpu")
    print(device)

    # data setting
    data_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    is_file = os.path.isfile(data_dir)
    single_image = None
    if (is_file):
        s_image = PIL.Image.open(data_dir)
        print(s_image)
        trs = data_transforms(s_image)
        single_image = SingleImage(data_dir, s_image, trs, image_size)

    dataloaders = None
    dataset_sizes = 1
    # load image from assigned file path

    if not is_file:
        image_datasets = datasets.ImageFolder(data_dir, data_transforms)

    # place image to dataloaders for pytorch model testing

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, num_workers=4)
    dataset_sizes = len(image_datasets)
    class_names = class_name_str.split("/")

    # model setting
    # model = models.resnet50(weights=None)
    # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    if (torch.cuda.is_available()):
        model.load_state_dict(torch.load(weight_path))
    else:
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    # summary(model)
    # print(model)
    # print(model)
    # summary(model,input_size=(3,224,224))

    # output
    # start_detect(device, class_names, model, dataloaders, dataset_sizes, show_image, single_image)
    return device, class_names, model, data_transforms


def detect_single_image_bk(device, class_names, model, single_image, data_transforms):
    original_image=single_image
    single_image = data_transforms(single_image)
    #single_image = SingleImage( s_image, single_image, 448, path="")
    single_image = SingleImage("",original_image, single_image, 448)
    model.eval()
    images = []
    imageInfo = ""
    count = 0

    with torch.no_grad():
        image = single_image.trans_image
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        outputs = model(image.to(device))
        _, preds = torch.max(outputs, 1)
        images.append(single_image.trans_image)
        imageInfo = class_names[preds[0]]

    return imageInfo


def detect_single_image(device, class_names, model, single_image):

    #single_image = SingleImage("",single_image, single_image, 448)
    model.eval()

    with torch.no_grad():
        #image = single_image
        image = Variable(single_image, requires_grad=True)
        image = image.unsqueeze(0)
        outputs = model(image.to(device))
        _, preds = torch.max(outputs, 1)
        #images.append(single_image)
        imageInfo = class_names[preds[0]]

    return imageInfo

predict_counter = 0
#print(predict_counter)
# data_dir='data/ToDetect/images/982889.jpg'
device, class_names, model, data_transforms = init_model()




@app.route('/test')
def predict_test():
    rslt = "hi"
    return rslt


@app.route('/fakepredict', methods=['POST'])
def fakepredict():
    if request.method == 'POST':
        files = request.files
        rslts = {}
        for file_name in files:
            rslts[file_name.split('/')[-1]] = "testOnly"
        print(rslts)
        return jsonify(rslts)





@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        rslts = {'msg': "ok", 'result': {}}
        try:
            files = request.files
            for file_name in files:


                shared_resource_lock.acquire()
                f_image = files[file_name].read()
                files[file_name].close()
                s_image = PIL.Image.open((io.BytesIO(f_image)))
                trans_image=data_transforms(s_image)
                s_image.close()
                #rslt = detect_single_image(device, class_names, model, s_image, data_transforms)
                s_image.close()
                rslt= detect_single_image(device, class_names, model, trans_image)
                shared_resource_lock.release()
                rslts['result'][file_name.split('/')[-1]] = rslt

                logging.info(datetime.now().strftime("[%Y/%m/%d %H.%M.%S] ") + file_name+": "+str(rslt))
                s_image.close()
            #print(rslts)
        except Exception as e:
            logging.error(datetime.now().strftime("[%Y/%m/%d %H.%M.%S] ") +str(e))
            rslts['msg'] = 'error'
        return jsonify(rslts)

# def hello():
#    return 'Hello World'
