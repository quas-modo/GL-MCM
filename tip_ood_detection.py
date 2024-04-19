import os
import argparse
import numpy as np
import torch
from scipy import stats
import yaml
import random

from datasets import build_ood_dataset
from datasets.imagenet import ImageNet
import clip
from datasets.utils import build_data_loader
from tip_utils import *
# from tip_utils import cal_criterion, cls_acc, cls_auroc_ours, clip_classifier, build_cache_model, pre_load_features

from utils.common import setup_seed, get_test_labels
from utils.detection_util import print_measures, get_and_print_results, get_ood_scores_clip
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import set_model_clip, set_val_loader, set_ood_loader_ImageNet


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates GL-MCM Score for out-of-domain target',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['ImageNet'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="./datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood",
                        type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type=int,
                        help='the GPU indice to use')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        help='mini-batch size')
    parser.add_argument('--T', type=int, default=1,
                        help='temperature parameter')
    parser.add_argument('--model', default='CLIP', type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/16', 'RN50', 'RN101'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='MCM', type=str, choices=['MCM', 'L-MCM', 'GL-MCM'], help='score options')
    parser.add_argument('--num_ood_sumple', default=-1, type=int, help="numbers of ood_sumples")
    parser.add_argument('--id_config', default="./configs/imagenet.yaml", help="in domain dataset settings in yaml format")
    parser.add_argument('--ood_config', default="./configs/inaturalist.yaml", help="out of domain dataset settings in yaml format")
    args = parser.parse_args()

    args.CLIP_ckpt_name = args.CLIP_ckpt.replace('/', '_')
    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt_name}_T_{args.T}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok=True)

    return args

def APE(log, cfg, cache_keys, cache_values,  test_features, test_labels, clip_weights, neg_clip_weights,
        open_features, open_labels):

    cfg['w'] = cfg['w_training_free']
    top_indices, ood_indices = cal_criterion(cfg, clip_weights, cache_keys, only_use_txt=True)

    top_cache_keys = cache_keys[top_indices, :]
    ood_cache_keys = cache_keys[ood_indices, :]
    new_cache_keys = torch.cat((top_cache_keys, ood_cache_keys), dim=1)

    new_test_features = test_features[:, top_indices]
    new_open_features = open_features[:, top_indices]

    new_cache_values = cache_values

    zero_clip_weights = torch.cat((clip_weights, neg_clip_weights), dim=1)
    clip_logits = 100. * test_features @ zero_clip_weights
    zero_shot_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zero_shot_acc))

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    affinity = new_test_features @ new_cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ new_cache_values
    tip_logits = clip_logits + cache_logits * alpha

    open_logits = 100. * open_features @ zero_clip_weights
    open_affinity = new_open_features @ new_cache_keys
    open_cache_logits = ((-1) * (beta - beta * open_affinity)).exp() @ new_cache_values
    open_tip_logits = open_logits + open_cache_logits * alpha

    auroc = cls_auroc_ours(tip_logits, open_tip_logits)
    log.debug("**** Our's test auroc: {:.2f}. ****\n".format(auroc))


def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)


    if args.in_dataset in ['ImageNet']:
        out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture']

    # Load configuration
    id_cfg = yaml.load(open(args.id_config, 'r'), Loader=yaml.Loader)
    ood_cfg = yaml.load(open(args.ood_config, 'r'), Loader=yaml.Loader)
    
    # Set cache
    id_cache_dir = os.path.join('/home/nfs03/zengtc/tip/caches', id_cfg['dataset'])
    os.makedirs(id_cache_dir, exist_ok=True)
    id_cfg['cache_dir'] = id_cache_dir

    log.debug("\nRunning in-domain dataset configs.")
    log.debug(id_cfg)

    clip_model, preprocess = set_model_clip(args)
    clip_model.eval()

    # ImageNet dataset
    random.seed(id_cfg['seed'])
    torch.manual_seed(id_cfg['seed'])

    log.debug("Preparing ImageNet dataset.")
    imagenet = ImageNet(id_cfg['root_path'], id_cfg['shots'], preprocess)

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

    # Textual features
    log.debug("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)
    neg_clip_weights = clip_classifier(imagenet.classnames, imagenet.neg_template, clip_model)

    # Construct the cache model by few-shot training set
    log.debug("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(log, id_cfg, clip_model, train_loader_cache)

    # Pre-load test features
    log.debug("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(id_cfg, "test", clip_model, test_loader)

    # Load open-set dataset
    log.debug("\nRunning out-domain dataset configs.")
    log.debug(ood_cfg)
    ood_cache_dir = os.path.join('/home/nfs03/zengtc/tip/caches', ood_cfg['dataset'])
    os.makedirs(ood_cache_dir, exist_ok=True)
    ood_cfg['cache_dir'] = ood_cache_dir
    ood_dataset = build_ood_dataset(ood_cfg['dataset'], ood_cfg['root_path'], log)
    ood_loader = build_data_loader(data_source=ood_dataset.all, batch_size=64, is_train=False, tfm=preprocess,
                                   shuffle=False)
    ood_features, ood_labels = pre_load_features(ood_cfg, "ood", clip_model, ood_loader)

    APE(log, id_cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, neg_clip_weights,
        ood_features, ood_labels)


if __name__ == '__main__':
    main()
