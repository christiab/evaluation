import cv2
import numpy as np
import os
#from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, jaccard_score, accuracy_score
import sklearn.metrics
from matplotlib import pyplot as plt
import argparse
import sys
from skimage.morphology import skeletonize, binary_dilation
from scipy.ndimage.filters import maximum_filter
import warnings
warnings.filterwarnings("ignore")
# Note:
# According to https://stackoverflow.com/a/22747902
# the warning
# 'libpng warning: iCCP: known incorrect sRGB profile'
# can be ignored


if __name__ == '__main__':
  # info: must both be grayscale (ground truth even binary) images of the same dimension. matching via path (one with jpg, one with png)
  parser = argparse.ArgumentParser()
  arg = parser.add_argument
  arg('--pred_path', type=str, required=True, help='path to the predicted probability maps')
  arg('--true_path', type=str, required=True, help='path to the ground truth labels')
  arg('--show_pr_curve', required=False, default=False, action='store_true', help='show the precision-recall curve')
  args = parser.parse_args()

  predictions = os.listdir(args.pred_path)
  print("Number of predictions: ", len(predictions))

  # global containers
  all_preds = np.array([])
  all_trues = np.array([])

  # load images
  for path in predictions:

    # load the predictions and the corresponding ground truth
    img_pred = cv2.imread(os.path.join(args.pred_path, path), cv2.IMREAD_COLOR)[:,:,2] / 255.0
    img_true = cv2.imread(os.path.join(args.true_path, path.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE) / 255.0

    # reduce prediction to skeleton
    img_pred_skel = img_pred.copy()
    img_pred_skel = skeletonize(img_pred_skel>0.0) * img_pred

    # select relevant class from ground truth
    # (non-crack=0, planking=0.8, crack=1)
    img_true = np.where(img_true>=0.9, 1.0, 0.0)

    # create tolerance mask
    kernel = np.array([[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]])
    img_true_tol = binary_dilation(img_true, selem=kernel)

    # apply tolerance to prediction
    img_pred_max = maximum_filter(img_pred, footprint=kernel)
    img_tmp = img_true * img_pred_max
    img_pred_tol = img_pred * (1-img_true_tol) + img_tmp

    # convert in sklearn.metrics compatible format
    all_trues = np.append(all_trues, img_true.flatten(), axis=0)  
    all_preds = np.append(all_preds, img_pred_tol.flatten(), axis=0)

    if False:
      plt.subplot(231)
      plt.imshow(img_pred)
      plt.subplot(232)
      plt.imshow(img_true)
      plt.subplot(233)
      plt.imshow(img_pred_skel)        
      plt.subplot(234)
      plt.imshow(img_true_tol)
      plt.subplot(235)
      plt.imshow(img_pred_max)    
      plt.subplot(236)
      plt.imshow(img_pred_tol)        
      plt.show() 

  # precision recall curve
  precision, recall, thresholds = sklearn.metrics.precision_recall_curve(all_trues, all_preds)
  
  # average precision
  ap = sklearn.metrics.average_precision_score(all_trues, all_preds, average='micro')
  print("AP: ", ap)

  # f1 score: loop over thresholds to find best f1
  f1 = []
  for thresh in thresholds[0:len(thresholds):100]:
    sys.stdout.write("\rThreshold Progress %f" % thresh)
    sys.stdout.flush()
    all_preds_tmp = all_preds.copy()
    all_preds_tmp = np.where(all_preds>=thresh, 1.0, 0.0)
    f1.append(sklearn.metrics.f1_score(all_trues, all_preds_tmp))

  f1 = np.array(f1)
  best_f1 = np.max(f1)
  best_thresh = thresholds[np.argmax(f1)*2]
  print("Best f1: ", best_f1)
  print("at threshold: ", best_thresh)

  # IoU (Jaccard index)
  iou = sklearn.metrics.jaccard_score(all_trues, all_preds>=best_thresh)
  print("IoU: ", iou)

  # accuracy
  acc = sklearn.metrics.accuracy_score(all_trues, all_preds>=best_thresh)
  print("ACC: ", acc)

  # plot precision recall curve
  if bool(args.show_pr_curve):
    # plot no skill
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.')
    # show the plot
    plt.show()
  

