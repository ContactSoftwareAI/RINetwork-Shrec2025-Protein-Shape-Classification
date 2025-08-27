"""
Author: Zhiyuan Zhang
Date: Dec 2021
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ProteinToPointCloudProcessorTest import DataProcessor

import trimesh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

import yaml
#from easydict import EasyDict
import pandas as pd


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', type=bool, default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='riconv2_cls_v2_largest', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=97, type=int, choices=[97],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='riconv_large_shrec_run1_8k_largest', help='experiment root')
    parser.add_argument('--use_normals', type=bool, default=True, help='use normals')
    parser.add_argument('--use_uniform_sample', type=bool, default=True, help='use uniform sampiling')
    return parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True



def test(model, loader, num_class=40, rot = False):
    classifier = model.eval()
    all_pred = []

    for j, points in tqdm(enumerate(loader), total=len(loader)):
        if rot == True:
            points = points[:,:,:6].data.numpy()
            points = provider.rotate_point_cloud_with_normal(points)
            points = torch.Tensor(points)

        if not args.use_cpu:
            points = points.cuda()

        pred, _ = classifier(points)
        if len(pred.shape) == 3:
            pred = pred.mean(dim=1)
        pred_choice = pred.data.max(1)[1].detach().cpu().numpy()
        #print(pred_choice.shape)
        for p in pred_choice:
            all_pred.append(p)

    return all_pred


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification_shrec2025')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    root_path = "./new_data/test_set_vtk/"
    data = pd.read_csv("./new_data/test_set_2.csv")
    filenames = data["anonymised_protein_id"]

    test_dataset = DataProcessor(root_path, filenames, split="test", pc_folder="/txt8/")
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    classifier = model.get_model(num_class, 2, normal_channel=args.use_normals)
    #classifier = PointBert()

    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth', weights_only = False)
    #checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])


    log_string('Trainable Parameters: %f' % (count_parameters(classifier)))



    with torch.no_grad():
        prediction = test(classifier.eval(), testDataLoader, num_class=num_class)

    print(len(prediction))
    data["predicted_label"] = prediction
    data.to_csv("test_set_2_run1.csv")





import time

if __name__ == '__main__':
    begin = time.time()
    args = parse_args()
    main(args)
    end = time.time()
    print(end - begin)
