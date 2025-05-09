import os
import argparse
from gorilla.config import Config
from os.path import join as opj
from utils import *
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Test model on unseen affordances")
    parser.add_argument("--config", help="config file path", default='config/openad_pn2/full_shape_open_vocab_cfg.py')
    parser.add_argument("--checkpoint", help="the dir to saved model", default='checkpoint/best_model_openad_pn2_estimation.t7')
    parser.add_argument(
        "--gpu",
        type=str,
        default=1,
        help="Number of gpus to use"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)

    logger = IOStream(opj(cfg.work_dir, 'result_' + cfg.model.type + '.log'))
    if cfg.get('seed', None) != None:
        set_random_seed(cfg.seed)
        logger.cprint('Set seed to %d' % cfg.seed)
        
    if args.gpu != None:
        cfg.training_cfg.gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    model = build_model(cfg).cuda()
    if args.checkpoint == None:
        print("Please specify the path to the saved model")
        exit()
    else:
        print("Loading model....")
        _, exten = os.path.splitext(args.checkpoint)
        if exten == '.t7':
            model.load_state_dict(torch.load(args.checkpoint))
        elif exten == '.pth':
            check = torch.load(args.checkpoint)
            model.load_state_dict(check['model_state_dict'])

    dataset_dict = build_dataset(cfg)
    loader_dict = build_loader(cfg, dataset_dict)
    val_loader = loader_dict.get("val_loader", None)
    val_affordance = cfg.training_cfg.val_affordance
    mIoU = evaluation(logger, model, val_loader, val_affordance)