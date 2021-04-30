import numpy as np
import torch
import torchvision
import argparse
import os
import math

from demo_dataset import DynamicDataset
from networks.depth_refinement_network import DRNModified
import matplotlib.pyplot as plt
from PIL import Image


def ParseCmdLineArguments():
    parser = argparse.ArgumentParser(description='Modified MARS CNN Script')
    parser.add_argument('--checkpoint', action='append',
                        default="checkpoints/Models/trained_model-0000030.cpkt",
                        help='Location of the checkpoints to evaluate.')
    parser.add_argument('--train', type=int, default=1,
                        help='If set to nonzero train the network, otherwise will evaluate.')
    parser.add_argument('--save', type=str, default='',
                        help='The path to save the network checkpoints and logs.')
    parser.add_argument('--save_visualization', type=str, default='',
                        help='Saving network output images.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--root', type=str, default='/mars/mnt/dgx/FrameNet')
    parser.add_argument('--epoch', type=int, default=0,
                        help='The epoch to resume training from.')
    parser.add_argument('--iter', type=int, default=0,
                        help='The iteration to resume training from.')
    parser.add_argument('--dataset_pickle_file', type=str, default='./pickles/test.pkl')
    parser.add_argument('--dataloader_test_workers', type=int, default=16)
    parser.add_argument('--dataloader_train_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1.e-4)
    parser.add_argument('--save_every_n_iteration', type=int, default=1000,
                        help='Save a checkpoint every n iterations (iterations reset on new epoch).')
    parser.add_argument('--save_every_n_epoch', type=int, default=1,
                        help='Save a checkpoint on the first iteration of every n epochs (independent of iteration).')
    parser.add_argument('--enable_multi_gpu', type=int, default=0,
                        help='If nonzero, use all available GPUs.')
    parser.add_argument('--skip_every_n_image_test', type=int, default=40,
                        help='Skip every n image in the test split.')
    parser.add_argument('--skip_every_n_image_train', type=int, default=1,
                        help='Skip every n image in the test split.')
    parser.add_argument('--eval_test_every_n_iterations', type=int, default=1000,
                        help='Evaluate the network on the test set every n iterations when in training.')
    parser.add_argument('--resnet_arch', type=int, default=18,
                        help='ResNet architecture for ModifiedFPN (18/34/50/101/152)')
    parser.add_argument('--surface_normal_checkpoint', type=str, default='',
                        help='Surface normal checkpoint path is a required field.')
    parser.add_argument('--predicted_normal_subdirectory', type=str, default='DORN_acos_bs16_inference',
                        help='Predicted surface normal subdir path is a required field.')
    parser.add_argument('--dataset_type', type=str, default='scannet',
                        help='The dataset loader fromat. Closely related to the pickle file (scannet, nyu, azure).')
    parser.add_argument('--max_epochs', type=int, default=10000,
                        help='Maximum number of epochs for training.')
    parser.add_argument('--depth_loss', type=str, default='L1',
                        help='Depth loss function: L1/L2')
    parser.add_argument('--metrics_averaged_among_images', type=int, default=1,
                        help='Which type of metric we are computing.')

    # Depth Refinement Network
    parser.add_argument('--drn_model', type=str, default='')
    parser.add_argument('--refinement_iterations', type=int, default=5,
                        help='Iterative refinement iterations')
    parser.add_argument('--uncertainty_threshold', type=float, default=-1.0,
                        help='Depth uncertainty thresholds.')
    return parser.parse_args()

def scaleInvariantUncertaintyLoss(gt, pred, uncertainty):
    # shifting and scaling to be within [0,1]
    mask = gt > 0
    if torch.sum(mask).item() == 0:
        return torch.sum(mask)
    gt_shifted = gt - torch.min(gt[mask])
    gt_scaled = gt_shifted / torch.max(gt_shifted[mask])
    pred_shifted = pred - torch.min(pred)
    pred_scaled = pred_shifted / torch.max(pred_shifted)
    uncertainty_shifted = uncertainty - torch.min(uncertainty)
    uncertainty_scaled = uncertainty_shifted / torch.max(uncertainty_shifted)

    # calculating loss
    diff = torch.abs(gt_scaled[mask] - pred_scaled[mask])
    loss = diff * torch.exp(-uncertainty_scaled[mask]) + uncertainty_scaled[mask]

    return torch.sum(loss) / gt.shape[0]

def scaleInvariantLoss(gt, pred):
    # shifting and scaling to be within [0,1]
    b = gt.shape[0]
    mask = gt > 0
    gt[torch.logical_not(mask)] = torch.mean(gt)  # 0 no longer contributes to min or max
    gt_shifted = (gt.T - gt.view(b,-1).min(axis=1).values).T
    gt_scaled = (gt_shifted.T / gt_shifted.view(b,-1).max(axis=1).values).T
    pred_shifted = (pred.T - pred.view(b,-1).min(axis=1).values).T
    pred_scaled = (pred_shifted.T / pred_shifted.view(b,-1).max(axis=1).values).T

    # calculating loss
    loss = torch.norm(gt_scaled[mask] - pred_scaled[mask])

    return loss

def missingPixelLoss(gt, pred, init):
    # shifting and scaling to be within [0,1]
    b = gt.shape[0]
    gt_shifted = (gt.T - gt.view(b,-1).min(axis=1).values).T
    gt_scaled = (gt_shifted.T / gt_shifted.view(b,-1).max(axis=1).values).T
    pred_shifted = (pred.T - pred.view(b,-1).min(axis=1).values).T
    pred_scaled = (pred_shifted.T / pred_shifted.view(b,-1).max(axis=1).values).T

    # calculating loss
    mask = init > 0
    loss = torch.norm(gt_scaled[mask] - pred_scaled[mask])

    return loss

if __name__ == "__main__":

    args = ParseCmdLineArguments()

    testset = DynamicDataset("train", args.dataset_pickle_file)
    testloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size,
			 	   shuffle = True, num_workers = 1)

    net = DRNModified(args)#.cuda()
    net.load_state_dict(torch.load(args.checkpoint))
    net.eval()

    running_loss = 0
    skipped = 0
    for i, data in enumerate(testloader, 0):
        # load data
        rgb = data['image']#.cuda(non_blocking=True)
        depth_gt = data['depth_gt']#.cuda(non_blocking=True)
        depth_init = data['predicted_depth']#.cuda(non_blocking=True)
        normal_pred = data['predicted_normal']#.cuda(non_blocking=True)
        uncertainty = data['uncertainty']#.cuda(non_blocking=True)
        output_path = data['output_path']

        # forward
        pred = net(rgb, normal_pred, depth_init, uncertainty)
        depth_pred = pred["d"][-1]
        uncertainty_pred = pred["u"][-1]
        loss = missingPixelLoss(depth_gt, depth_pred, depth_init)
        if math.isnan(loss.item()) or loss.item() == 0:
            skipped += 1
        else:
            running_loss += loss.item()

        # save output
        for j in range(len(output_path)):    
            depth_scaled = depth_pred[j,0] - torch.min(depth_pred[j,0])
            #depth_array = (255 * depth_scaled / torch.max(depth_scaled)).cpu().detach().numpy()
            depth_array = (255 * depth_scaled / torch.max(depth_scaled)).detach().numpy()
            img = Image.fromarray(depth_array.astype(np.uint8), 'L')

            if not os.path.exists(os.path.dirname(output_path[j])):
                os.makedirs(os.path.dirname(output_path[j])) 
            img.save(output_path[0])

            if loss.item() == 0:
                print("***", output_path[j])
            else:
                print(output_path[j])

    print("Average loss: {:.2f}".format(running_loss / (len(testset) - skipped)))

