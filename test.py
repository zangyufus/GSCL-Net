import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from dataset import PartNormalDataset
import hydra
import omegaconf

import random
import colorsys

import sklearn.metrics as metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

seg_classes =  {'AREA1': [0, 1, 2, 3, 4, 5, 6, 7]}

seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def export_pc(points, colors, true_ground, filename):
    num_points = points.shape[0]
    if colors is not None:
        vertices = np.empty(num_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('target', 'i4')])
        vertices['x'] = points[:,0].astype('f4')
        vertices['y'] = points[:,1].astype('f4')
        vertices['z'] = points[:,2].astype('f4')
        vertices['red'] = colors[:,0].astype('u1')
        vertices['green'] = colors[:,1].astype('u1')
        vertices['blue'] = colors[:,2].astype('u1')
        vertices['target'] = true_ground.astype('i4')
        with open(filename, 'w') as f:
            f.write("x,y,z,red,green,blue,target\n")
            for i in range(num_points):
                x, y, z, r, g, b, t = vertices[i]
                f.write("{},{},{},{},{},{},{}\n".format(x, y, z, r, g, b, t))
    else:
        vertices = np.empty(num_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('target', 'i4')])
        vertices['x'] = points[:,0].astype('f4')
        vertices['y'] = points[:,1].astype('f4')
        vertices['z'] = points[:,2].astype('f4')
        vertices['t'] = true_ground.astype('i4')
        with open(filename, 'w') as f:
            f.write("x,y,z,target\n")
            for i in range(num_points):
                x, y, z, t = vertices[i]
                f.write("{},{},{},{}\n".format(x, y, z, t))

def visualize_segmentation(pc, color, cur_pred_val_logits, true_ground, save_path, bias, names):
    batch_size = pc.shape[0]
    pc = pc.cpu().detach().numpy()
    true_ground = true_ground.numpy()
    for k in range(batch_size):
        if names is not None:
            filename = save_path + names[k] + '_segment.txt'
        else:
            filename = save_path + str(bias + k) + '_segment.txt'
        color_segment = color[np.argmax(cur_pred_val_logits[k,:,:],1),:]
        export_pc(pc[k, :, :], color_segment, true_ground[k, :], filename)

def generate_ncolors(num):
    def get_n_hls_colors(num):
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            _hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(_hlsc)
            i += step
        return hls_colors
    rgb_colors = np.zeros((0,3))
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors = np.concatenate((rgb_colors,np.array([r,g,b])[np.newaxis,:]))
    return rgb_colors


@hydra.main(config_path='config', config_name='partseg', version_base=None)
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    # print(args.pretty())

    root = hydra.utils.to_absolute_path('D:/YYY-LSK/YYY_new/data/nusit_data_sampling/')

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0)

    '''MODEL LOADING'''
    args.num_class = 8
  
    num_category = 1
    args.input_dim = 8 + num_category
    
    num_part = args.num_class
    color = generate_ncolors(num_part)
   

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    

    try:
        checkpoint = torch.load('best_model16.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')

    except:
        logger.info('No existing model, exiting...')
        sys.exit(0)




    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        accuracy_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}

        # total_iou_deno_class = [0 for _ in range(num_part)]
        # total_iou_num_class = [0 for _ in range(num_part)]
        total_seen = 0

        save_path = 'D:\\YYY-LSK\\YYY_new\\Point-Transformers-master_PCT\\log\\JND-test-16 7.13\\'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        test_true_cls = []
        test_pred_cls = []



        for batch_id, (points, originxyz, label, target, cur_name) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            xyz = originxyz[:, :, :3]
            true_ground = target
            points, label, target= points.float().cuda(), label.long().cuda(), target.long().cuda()
            seg_pred = classifier(
                torch.cat([points, to_categorical(label, num_category).repeat(1, points.shape[1], 1)], -1))


            cur_pred_val = seg_pred.cpu().data.numpy() #[2, 2048, 6]
            cur_pred_val_logits = cur_pred_val #[2, 2048, 6]
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32) #[2, 2048]
            target = target.cpu().data.numpy()

            #seg_pred = seg_pred.permute(0, 2, 1).contiguous() #[2, 6, 2048]
            pred = seg_pred.max(dim=2)[1] # [2, 6]
            seg_np = target
            pred_np = pred.detach().cpu().numpy() #(2, 2048)
            test_true_cls.append(seg_np.reshape(-1)) # shape(4096,)
            test_pred_cls.append(pred_np.reshape(-1)) # shape(4096,)

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))
                accuracy_class[l] = total_correct_class[l] / float(total_seen_class[l])


            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))] #cat:AREA1

                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))
            visualize_segmentation(xyz, color, cur_pred_val_logits, true_ground, save_path, _, cur_name)

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)

        print("--------accuracy----------")
        print(test_metrics['accuracy'])

        # test_metrics['class_avg_accuracy'] = np.mean(
        #     np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        # logger.info('accuracy' % (test_metrics['accuracy']))
        for l in range(num_part):
            logger.info('eval accuracy of %d %f' % (l , accuracy_class[l]))

        for cat in sorted(shape_ious.keys()):
            logger.info('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)


        test_true_cls = np.concatenate(test_true_cls) #shape(1749600,)
        test_pred_cls = np.concatenate(test_pred_cls)
        conf_mat = metrics.confusion_matrix(test_true_cls, test_pred_cls)
        print('Confusion matrix:')
        print(str(conf_mat))

        iou_list = []
        for class_id in range(conf_mat.shape[0]):
            TP = conf_mat[class_id, class_id]
            FP = np.sum(conf_mat[:, class_id]) - TP
            FN = np.sum(conf_mat[class_id, :]) - TP
            IoU = TP / (TP + FP + FN)
            iou_list.append(IoU)
        for class_id, iou in enumerate(iou_list):
            print(f'Class {class_id} IoU: {iou:.4f}')
            # logger.info('Class %d %f IoU:' % (iou))
        print(f'mIoU {np.mean(iou_list)}')


if __name__ == '__main__':
    main()
