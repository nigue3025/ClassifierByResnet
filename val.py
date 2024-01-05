from __future__ import print_function, division

import sys

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
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.classification import MulticlassPrecisionRecallCurve
import random
import argparse
import json


def tensorToNpImage(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp
    # plt.imshow(inp)
    # if title is not None:
    #    plt.title(title)
    # plt.pause(1000)
    # plt.pause(1000)  # pause a bit so that plots are updated


def evaluate_model(class_names, dataloaders, model, device):
    # was_training = model.training
    model.eval()
    images_so_far = 0
    # initialize ground truth count and accuracy table
    correctCount = {}
    accu = {}
    for name in class_names:
        correctCount[name] = 0
        accu[name] = 0
    accu['overall'] = 0
    # initialize confusion matrix
    confsMatrx = {}
    for x in class_names:
        confsMatrx[x] = {}
        for y in class_names:
            confsMatrx[x][y] = 0
    # initialize accuracy table
    labelLst = []
    predctLst = []
    # initialize image to show container
    images = []
    imageInfo = []
    output_softmax=[]
    # testdataloader=dataloaders['test']
    for i, (inputs, labels) in enumerate(dataloaders['test']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            labelClass = class_names[labels[j]]
            predClass = class_names[preds[j]]
            output_softmax.append(float(_[j]))
            labelLst.append(int(labels[j]))
            predctLst.append(int(preds[j]))
            correctCount[labelClass] = correctCount[labelClass] + 1
            confsMatrx[labelClass][predClass] = confsMatrx[labelClass][predClass] + 1
            images.append(inputs.cpu().data[j])
            filename = dataloaders['test'].dataset.imgs[images_so_far - 1][0].split('/')[-1]
            imageInfo.append(
                {
                'filename': filename,
                 'predict': class_names[preds[j]],
                 'actual': class_names[labels[j]],
                 'correct': bool(labels[j] == preds[j]),
                 'info': f'{filename}\npredict: {class_names[preds[j]]}\nactual:{class_names[labels[j]]}'
                 })

    # compute f1,precision,recall
    #Due to labeling error (Blurry should be marked as 1 instead), use 1-x for correcting f1,recall,precision calculation
    f1 = BinaryF1Score()
    f1 = f1(torch.tensor([1-x for x in predctLst]), torch.tensor([1-x for x in labelLst]))
    br = BinaryRecall()
    br = br(torch.tensor([1-x for x in predctLst]), torch.tensor([1- x for x in labelLst]))
    bp = BinaryPrecision()
    bp = bp(torch.tensor([1-x for x in predctLst]), torch.tensor([1-x for x in labelLst]))

    bprc=BinaryPrecisionRecallCurve(thresholds=None)
    precision,recall,thresholds=bprc(torch.tensor(output_softmax), torch.tensor([x for x in labelLst]))
    bprc.compute()
    #mrc=MulticlassPrecisionRecallCurve(num_classes=2)
    #targets=torch.tensor([1,0])
    #precision,recall,thresholds=mrc(torch.tensor(output_softmax),torch.tensor(labelLst))
    precision=precision.tolist()
    recall=recall.tolist()




    plt.plot(recall,precision,label='aaa')
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-recall Curve (Resnet50 for blurry detection)")
    plt.show()
    #plt.plot(thresholds.tolist())
    #plt.title("Sorted Thresholds")
    #plt.ylabel("Threshold")
    #plt.show()

    # compute accuracy
    ccount = 0
    total = 0
    for name in class_names:
        accu[name] = confsMatrx[name][name] / correctCount[name]
        ccount = ccount + confsMatrx[name][name]
        total = total + sum(confsMatrx[name].values())
    print(sum(confsMatrx['Blurry'].values()))
    accu['overall'] = float(ccount) / float(total)
    eval_data = {'accuracy': accu, 'recall score': float(br), 'precision score': float(bp), 'f1 score': float(f1),
                 'confusion matrix': confsMatrx}
    return eval_data, images, imageInfo,bprc


def display_result(images, image_info, bprc, display_num=12):
    indices = [x for x in range(0, len(images))]
    random.shuffle(indices)
    # display images
    columns = 4
    rows = 3
    fig = plt.figure(figsize=(8, 8))
    axs = []
    for i in range(0, display_num):
        rnd_index = indices[i]
        ax = fig.add_subplot(rows, columns, i + 1)
        axs.append(ax)
        img = tensorToNpImage(images[rnd_index])
        filename = image_info[rnd_index]['filename']
        if image_info[rnd_index]['correct']:
            ax.set_title(image_info[rnd_index]['info'], size=12, color='black')
        else:
            ax.set_title(image_info[rnd_index]['info'], size=12, color='red', fontweight="bold")
        ax.axis('off')
        plt.imshow(img)
    plt.show()

    plt.close()
    print(bprc)
    plt.show()


def generate_new_testfolder():
    num = 1
    while os.path.exists("TestResult/Test"+str(num)):
        num += 1
    newdir="TestResult/Test"+str(num)
    os.mkdir(newdir)
    return newdir


def write_result(eval_data, image_info):
    json_object=json.dumps(eval_data)
    json_object = json.dumps(image_info)
    newdir = generate_new_testfolder()

    with open(newdir+"/rslt.json","w") as outfile:
        outfile.write((json_object))
    predictions = {}
    for info in image_info:
        if not info['predict'] in predictions:
            predictions[info['predict']]=[]
        predictions[info['predict']].append(info['filename'])
    json_object = json.dumps((predictions))
    with open(newdir+"/prediction.json","w") as outfile:
        outfile.write((json_object))

    json_object = json.dumps((eval_data))
    with open(newdir+"/test_evaluation.json", "w") as outfile:
        outfile.write((json_object))


def test(selected_device='gpu', weight_path='models/pytorch/resnet50_adam_20epochs_weights.h5',
         data_dir='data/BlurryMarkupAll/Sampling', batch_size=1, show_image=False):
    # device setting        # visualize_model(device, class_names, model, dataloaders, dataset_sizes, num_images=10)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if selected_device == 'cpu':
        device = torch.device("cpu")
    print(device)
    # data setting
    # do not add random factor to testing data set (e.g. transforms.RandomResizedCrop(224))
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    # load image from assigned file path
    image_datasets = {'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])}

    # place image to dataloaders for pytorch model testing
    dataloaders = {'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, num_workers=4,
                                                       shuffle=False)}
    class_names = image_datasets['test'].classes
    # model setting
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path))
    eval_data, images, image_info,bprc = evaluate_model(class_names, dataloaders, model, device)
    print(eval_data)
    write_result(eval_data, image_info)

    if show_image:
        display_result(images, image_info,bprc, 12)


if __name__ == '__main__':
    defaultModelPath = 'models/pytorch/resnet50_adam_20epochs_weights.h5'
    defaultDataPath = 'data/BlurryMarkupAll/Sampling'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', required=False, type=int, default=8, help='batch size')
    parser.add_argument('--weight-path', required=False, type=str, default=defaultModelPath,
                        help='path of model weight')
    parser.add_argument('--data-path', required=False, type=str, default=defaultDataPath,
                        help='path of testing data folder')
    parser.add_argument('--device', required=False, type=str, default="gpu", help='gpu or cpu')
    parser.add_argument('--show-image', required=False, type=bool, default=True,
                        help='True or False (sample 12 images to display')
    args = parser.parse_args()
    test(selected_device=args.device,
         batch_size=args.batch_size,
         weight_path=args.weight_path,
         data_dir=args.data_path,
         show_image=args.show_image
         )
