"""
Run detection on all images and save detected bounding box (with category_name and score).

cache/detections/refcoco_unc/{net}_{imdb}_{tag}_dets.json has
0. dets: list of {det_id, box, image_id, category_id, category_name, score}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import numpy as np
import argparse
import json
import torch
from scipy.misc import imread

# add paths
import _init_paths
from model.nms_wrapper import nms
from mrcn import inference, inference_no_imdb
import pickle

""" CHANGE CONFIG BETWEEN RUNNING PROPOSE !!!
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000  # 4000 to extract RPN proposals

# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300  # 400 to extract RPN proposals
"""

def main(args):

  # Image Directory
  params = vars(args)
  dataset_splitBy = params['dataset'] + '_' + params['splitBy']
  if 'coco' or 'combined' in dataset_splitBy:
    IMAGE_DIR = 'data/images/mscoco/images/train2014'
  elif 'clef' in dataset_splitBy:
    IMAGE_DIR = 'data/images/saiapr_tc-12'
  else:
    print('No image directory prepared for ', args.dataset)
    sys.exit(0)

  # make save dir
  save_dir = osp.join('cache/detections', dataset_splitBy)
  if not osp.isdir(save_dir):
    os.makedirs(save_dir)
  print(save_dir)

  # get mrcn instance
  mrcn = inference.Inference(args)
  imdb = mrcn.imdb

  # import refer
  from refer import REFER
  data_root, dataset, splitBy = params['data_root'], params['dataset'], params['splitBy']
  refer = REFER(data_root, dataset, splitBy)
  cat_name_to_cat_ix = {category_name: category_id for category_id, category_name in refer.Cats.items()}

  # detect and prepare dets.json
  proposals = []
  det_id = 0
  cnt = 0

  # # TEMPS DEBUG
  # # os.makedirs('cache/old_internals')
  # img_path = '/home/mwb/Datasets/mscoco/images/train2014/COCO_train2014_000000581857.jpg'
  # scores, boxes = mrcn.predict(img_path)
  # image_feat = mrcn.net._predictions['__temp_net_conv'].data.cpu().numpy()
  # roi_feats = mrcn.net._predictions['__temp_pool5'].data.cpu().numpy()
  # rois = mrcn.net._predictions['__temp_rois'].data.cpu().numpy()[:,1:]
  # head_feats = mrcn.net._predictions['__temp_head_feats'].data.cpu().numpy()
  # head_pool = mrcn.net._predictions['__temp_head_pool'].data.cpu().numpy()
  # print(image_feat.shape, roi_feats.shape, rois.shape, head_feats.shape, head_pool.shape)
  # np.save('cache/old_internals/image_feat.npy', image_feat)
  # np.save('cache/old_internals/roi_feats.npy', roi_feats)
  # np.save('cache/old_internals/rois.npy', rois)
  # np.save('cache/old_internals/head_feats.npy', head_feats)
  # np.save('cache/old_internals/head_pool.npy', head_pool)

  for image_id, image in refer.Imgs.items():
    file_name = image['file_name']
    img_path = osp.join(IMAGE_DIR, file_name)

    # predict
    scores, boxes = mrcn.predict(img_path)

    rois = mrcn.net._predictions['rois'].data.cpu().numpy()[:,1:] / mrcn._scale
    cnt += 1
    print('%s/%s done.' % (cnt, len(refer.Imgs)))

    info = {
      'image_id': image_id,
      'rois': rois,
      'scores': scores, 
      'boxes': boxes,
      'roi_scores': mrcn.net._predictions['__roi_scores'].data.cpu().numpy()
    }
    torch.cuda.empty_cache()

    proposals.append(info)
    
  # save dets.json = [{det_id, box, image_id, score}]
  # to cache/detections/
  save_path = osp.join(save_dir, '%s_%s_%s_proposals.pkl' % (args.net_name, args.imdb_name, args.tag))
  with open(save_path, 'wb') as f:
    pickle.dump(proposals, f)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_root', default='data', type=str, help='data folder containing images and four datasets.')
  parser.add_argument('--dataset', default='refcoco', type=str, help='refcoco/refcoco+/refcocog')
  parser.add_argument('--splitBy', default='unc', type=str, help='unc/google')

  parser.add_argument('--imdb_name', default='coco_minus_refer', help='image databased trained on.')
  parser.add_argument('--net_name', default='res101')
  parser.add_argument('--iters', default=1250000, type=int)
  parser.add_argument('--tag', default='notime')

  parser.add_argument('--nms_thresh', default=0.3, help='NMS threshold')
  parser.add_argument('--conf_thresh', default=0.65, help='confidence threshold')

  args = parser.parse_args()

  main(args)


