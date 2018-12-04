import argparse
import scipy.io
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from model import ft_net, ft_net_dense, PCB, PCB_test
import torch.nn as nn
import cv2
import random

#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=2, type=int, help='test_image_index')
parser.add_argument('--test_dir', default='./Friends', type=str, help='./test_data')
opts = parser.parse_args()
opt = opts
opt.use_dense = False
opt.PCB = False

#######################################################################
# Load data
data_dir = opts.test_dir
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['gallery', 'query']}

data_transforms = transforms.Compose([
        transforms.Resize((288, 144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

folder_path = './Friends\ep1\gallery/'
folder_name = os.listdir(folder_path)

#####################################################################
# Show result

def imshow(path, title):
    """Imshow for Tensor."""
    im = plt.imread(path)
    # plt.figure()
    plt.imshow(im)
    plt.title(title)
    plt.show()

    plt.axis('off')
    # if title is not None:

    # plt.pause(0.1)  # pause a bit so that plots are updated

def imageshow(path, title=None):
    im = cv2.imread(path)
    cv2.imshow(title, im)
    cv2.moveWindow(title, 500, 500)
    cv2.waitKey()
    # cv2.destroyWindow(title)

######################################################################
# Load model

def load_network(network):
    save_path = './model/ft_ResNet50/net_last.pth'
    network.load_state_dict(torch.load(save_path))
    return network

######################################################################
# Feature extract

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        # print(count)
        if opt.use_dense:
            ff = torch.FloatTensor(n, 1024).zero_()
        else:
            ff = torch.FloatTensor(n, 2048).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_()  # we have six parts
        for i in range(2):
            if(i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff+f
        # norm feature
        if opt.PCB:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)
    return features

if opt.use_dense:
    model_structure = ft_net_dense(751)
else:
    model_structure = ft_net(751)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if not opt.PCB:
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()
else:
    model = PCB_test(model)

# Change to test mode
model = model.eval()
model = model.cuda()

file_path = '../friends2/'
file_name = os.listdir(file_path)
file_name.sort()

# Bounding-box colors
colors = []

for c in range(20):
    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    colors.append(color)

# Bounding box data
gt_bbox = open('./friends2_gt.txt', 'r')
object_num = scipy.io.loadmat('./object_num.mat')
object_num = object_num['object_num'][0]

txt_position = 20

previous_id = 0

# Visualization
for v in range(129):

    if v == 0:
        image_datasets = datasets.ImageFolder(
            './Friends\ep1/gallery/' + folder_name[v],
            data_transforms)
        dataloaders_q = torch.utils.data.DataLoader(image_datasets)

        # Extract feature
        query_feature = extract_feature(model, dataloaders_q)

        current_id = list(range(1, query_feature.size()[0] + 1))
    else:

        # Gallery
        image_datasets = datasets.ImageFolder(
            './Friends\ep1/gallery/' + folder_name[v-1],
            data_transforms)
        dataloaders_g = torch.utils.data.DataLoader(image_datasets)

        # Query
        image_datasets = datasets.ImageFolder(
            './Friends\ep1/gallery/' + folder_name[v],
            data_transforms)
        dataloaders_q = torch.utils.data.DataLoader(image_datasets)

        # Extract feature
        gallery_feature = extract_feature(model, dataloaders_g)
        query_feature = extract_feature(model, dataloaders_q)

        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()

        current_id = list(range(1, query_feature.size()[0] + 1))

        score = []
        for x in range(query_feature.size()[0]):
            query = query_feature[x].view(-1, 1)
            score_value = torch.mm(gallery_feature, query)
            score_value = score_value.cpu().numpy()
            score.append(score_value)

        # return max index
        max_index = np.argmax(score, axis=1)

        for idx in range(query_feature.size()[0]):
            max_idx = int(max_index[idx])
            current_id[idx] = previous_id[max_idx]

    full_scene = cv2.imread(file_path+file_name[v])

    person_num = object_num[v]
    for num_person in range(person_num):
        gt_line = gt_bbox.readline()
        bbox_data = gt_line.split()
        bbox = bbox_data[2:6]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = x1 + int(int(bbox[2]))
        y2 = y1 + int(int(bbox[3]))

        id_number = int(current_id[num_person])
        id_color = colors[id_number]
        cv2.rectangle(full_scene, (x1, y1), (x2, y2), id_color, 6)
        cv2.putText(full_scene, 'Id: ' + str(current_id[num_person]), (x1 - txt_position, y1 - txt_position), cv2.FONT_HERSHEY_SIMPLEX, 1, id_color, 4, cv2.LINE_AA)
        resize_img = cv2.resize(full_scene, (960, 540))
        cv2.imshow('Friends', resize_img)
        cv2.waitKey(1)

    previous_id = current_id

cv2.waitKey(0)

