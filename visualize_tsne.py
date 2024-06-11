import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from main_ce import set_loader
from networks.resnet_big import SupConResNet

def parse_option():
    parser = argparse.ArgumentParser('argument for visualization')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='number of workers for data loader')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'STL10'], help='dataset')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    if opt.dataset == 'cifar10' or opt.dataset == 'STL10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def set_model(opt):
    model = SupConResNet(name='resnet50')
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
        model.head = None
    else:
        raise NotImplementedError('This code requires GPU')

    return model

def extract_features(model, loader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for (images, target) in loader:
            images = images.cuda()
            feat = model.encoder(images)
            features.append(feat.cpu().numpy())
            labels.append(target.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def plot_tsne(features, labels, n_cls):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 10))
    for i in range(n_cls):
        idx = labels == i
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], s=5)
    plt.axis('off')  
    plt.show()
    plt.savefig("CIFAR-10-AWCL.png", format='png', dpi=300)

def main():
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model
    model = set_model(opt)

    # extract features
    features, labels = extract_features(model, val_loader)

    # plot tsne
    plot_tsne(features, labels, opt.n_cls)

if __name__ == '__main__':
    main()
