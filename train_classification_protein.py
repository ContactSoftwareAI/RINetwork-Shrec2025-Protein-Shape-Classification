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
import pandas as pd

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ProteinToPointCloudProcessor import DataProcessor

import trimesh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', type=bool, default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='riconv2_cls_v2_largest', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=97, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=250, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default='riconv_large_shrec_run1_8k_largest', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', type=bool, default=True, help='use normals')
    parser.add_argument('--process_data', type=bool, default=True, help='save data offline')
    parser.add_argument('--use_uniform_sample', type=bool, default=True, help='use uniform sampiling')
    return parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True



def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        pred, _ = classifier(points)
        if len(pred.shape) == 3:
            pred = pred.mean(dim=1)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


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
    root_path = ".../shrec2025/new_data/train_set_vtk/"
    data = pd.read_csv(".../shrec2025/new_data/train_set_2.csv")
    filenames = data["protein_id"]
    labels = data["class_id"]

    train_data = filenames[:7428]
    train_labels = labels[:7428]

    validation_data = filenames[7428:]
    validation_labels = labels[7428:]

    train_dataset = DataProcessor(root_path, train_data, train_labels, pc_folder="/txt8/")
    test_dataset = DataProcessor(root_path, validation_data, validation_labels, pc_folder="/txt8/")


    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/%s_utils.py' % args.model.split('_')[0], str(exp_dir))

    class_weights = np.histogram(train_labels, bins=97)[0]
    class_weights = 1 / class_weights
    class_weights = torch.Tensor(class_weights).float()
    print(class_weights.shape)

    classifier = model.get_model(num_class, 2, normal_channel=args.use_normals)
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    #model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()



    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_instance_acc = checkpoint['instance_acc']
        best_class_acc = checkpoint['class_acc']
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
    #pytorch_total_params = sum(p.numel() for p in model.parameters())
    #print(pytorch_total_params)

    log_string('Trainable Parameters: %f' % (count_parameters(classifier)))

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points[:,:,:6].data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            
            if len(pred.shape) == 3:
                target_2 = target.unsqueeze(-1).repeat(1,pred.shape[1])
                target_2 = target_2.view(-1, 1)[:, 0]
                pred_2 = pred.contiguous().view(-1, num_class)  # N*K, num_class
                loss = criterion(pred_2, target_2.long())
                pred = pred.mean(dim=1) # N, num_class
            else:
                loss = criterion(pred, target.long())

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


import time
if __name__ == '__main__':
    begin = time.time()
    args = parse_args()
    main(args)
    end = time.time()
    print(end - begin)
