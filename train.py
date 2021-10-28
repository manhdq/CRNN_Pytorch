import argparse
import random
import os
import sys
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from datasets.dataset import randomSequentialSampler

from models import CRNN
from utils import Visualizer, strLabelConverter, averager, PolyLR, loadData
from datasets import textDataset, resizeNormalize, alignCollate


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDir', required=True, help='path to dataset')
    parser.add_argument('--trainCSV', required=True, help='csv file for train data')
    parser.add_argument('--valDir', required=True, help='path to dataset')
    parser.add_argument('--valCSV', required=True, help='csv file for val data')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrained', default='', help='path to pretrained model (to continue training)')
    parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='input batch size')
    parser.add_argument('--expr-dir', default='expr', help='where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--n-test-disp', type=int, default=10, help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadelta')
    parser.add_argument('--lr-policy', type=str, default='poly', choices=['poly', 'step'], help='learning rate scheduler policy')
    parser.add_argument('--step-size', type=int, default=10000)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default 0.5')
    parser.add_argument('--adam', action='store_true', help='whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true', help='whether to use adadelta (default is rmsprop)')
    parser.add_argument('--keep-ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--manualSeed', type=int, default=2103, help='reproduce experiment')
    parser.add_argument('--random-sample', action='store_true', help='whether to sample the dataset with random sampler')
    parser.add_argument('--continue-training', action='store_true', default=False)
    
    # Visdom options
    parser.add_argument('--enable-vis', action='store_true', help='use visdom for visualization')
    parser.add_argument('--vis-port', type=str, default='8097', help='port for visdom')
    parser.add_argument('--vis-env', type=str, default='main', help='env for visdom')
    parser.add_argument('--vis-num-samples', type=int, default=8, help='number of samples for visualization. default 8')

    return parser


def main():
    opts = get_argparser().parse_args()
    print(opts)

    # Setup visualization
    vis = Visualizer(opts.vis_port, opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table('Options', vars(opts))

    if not os.path.exists(opts.expr_dir):
        os.makedirs(opts.expr_dir)

    random.seed(opts.manualSeed)
    np.random.seed(opts.manualSeed)
    torch.manual_seed(opts.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opts.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    train_dataset = textDataset(opts.trainDir, opts.trainCSV)
    assert train_dataset
    if opts.random_sample:
        sampler = randomSequentialSampler(train_dataset, opts.batch_size)
    else:
        sampler = None
    
    train_loader = DataLoader(train_dataset,
                            batch_size=opts.batchSize,
                            shuffle=True, sampler=sampler,
                            num_workers=int(opts.workers),
                            collate_fn=alignCollate(imgH=opts.imgH, imgW=opts.imgW, keep_ratio=opts.keep_ratio))
    test_dataset = textDataset(opts.valDir, opts.valCSV, transform=resizeNormalize((100, 32)))

    nclass = len(opts.alphabet) + 1
    nc = 1

    converter = strLabelConverter(opts.alphabet)
    criterion = nn.CTCLoss()

    # Custom weights initialization called on crnn
    def weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif class_name.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    
    crnn = CRNN(opts.imgH, nc, nclass, opts.nh)
    crnn.apply(weights_init)

    image = torch.FloatTensor(opts.batchSize, 3, opts.imgH, opts.imgH)
    text = torch.IntTensor(opts.batchSize * 5)
    length = torch.IntTensor(opts.batchSize)

    # Loss averager
    loss_avg = averager()

    # Setup optimizer
    if opts.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))
    elif opts.adadelta:
        optimizer = optim.Adadelta(crnn.parameters())
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=opts.lr)

    if opts.lr_policy == 'poly':
        scheduler = PolyLR(optimizer, opts.nepoch * len(train_loader), power=.9)
    elif opts.lr_policy == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=.1)

    best_acc = 0.0
    cur_itrs = 0

    if opts.pretrained != '':
        checkpoint = torch.load(opts.pretrained, map_location=torch.device('cpu'))
        crnn.load_state_dict(checkpoint['model_state'])
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            cur_itrs = checkpoint["cur_itrs"]
            best_acc = checkpoint['best_acc']
            print('Training state restored from %s' % opts.pretrained)
        print('Model restored from %s' % opts.pretrained)
        del checkpoint

    print(crnn)

    if opts.cuda:
        crnn.cuda()
        crnn = nn.DataParallel(crnn, device_ids=range(opts.ngpu))
        image = image.cuda()
        criterion = criterion.cuda()

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    def save_ckpt(path):
        """Save current model"""
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": crnn.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_acc": best_acc,
        }, path)
        print('Model saved as %s' % path)


    def val(net, dataset, criterion, max_iter=100):
        print('Start val...')
        nonlocal best_acc

        for p in crnn.parameters():
            p.requires_grad = False

        net.eval()
        data_loader = DataLoader(dataset, shuffle=True, batch_size=opts.batchSize, num_workers=int(opts.workers))
        val_iter = iter(data_loader)

        i = 0
        n_correct = 0
        loss_avg = averager()

        max_iter = min(max_iter, len(data_loader))
        for i in range(max_iter):
            data = val_iter.next()
            i += 1
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts)
            loadData(text, t)
            loadData(length, l)

            preds = crnn(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            cost = criterion(preds, text, preds_size, length)
            loss_avg.add(cost)

            _, preds = preds.max(2)
            # preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, cpu_texts):
                if pred.lower() == target.lower():
                    n_correct += 1

        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opts.n_test_disp]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

        accuracy = n_correct / float(max_iter * opts.batchSize)
        print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))

        if vis is not None:
            vis.vis_scalar("[Val] Loss", cur_itrs, float(loss_avg.val()))
            vis.vis_scalar("[Val] Accuracy", cur_itrs, accuracy)

        if accuracy > best_acc:
            print(f'Accuracy improved ({best_acc} -> {accuracy}). Saving new best model .......')
            save_ckpt('{0}/best_netCRNN.pth'.format(opts.expr_dir))
            best_acc = accuracy


    def trainBatch(net, criterion, optimizer, scheduler):
        data = train_iter.next()
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        loadData(text, t)
        loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length)
        crnn.zero_grad()
        cost.backward()
        optimizer.step()
        scheduler.step()
        return cost


    for epoch in range(opts.nepoch):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()

            cost = trainBatch(crnn, criterion, optimizer, scheduler)
            loss_avg.add(cost)
            i += 1
            cur_itrs += 1
            
            if i % opts.displayInterval == 0:
                if vis is not None:
                    vis.vis_scalar('Loss', cur_itrs, float(loss_avg.val()))

                print('[%d/%d][%d/%d] Loss: %f' %
                    (epoch, opts.nepoch, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            if i % opts.valInterval == 0:
                val(crnn, test_dataset, criterion)

            # do checkpointing
            if i % opts.saveInterval == 0:
                save_ckpt('{0}/lastest_CRNN.pth'.format(opts.expr_dir))


if __name__ == '__main__':
    main()