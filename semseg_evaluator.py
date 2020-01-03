import cv2
import numpy as np
import os
#from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, jaccard_score, accuracy_score
import sklearn.metrics
from matplotlib import pyplot as plt
import argparse
import sys

#pred_path = "/home/chrisbe/repos/crack_segmentation/test_result_23"
#true_path = "/home/chrisbe/repos/crack_segmentation/datasets/DeepCrack/test_lab"

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
  arg('--show_pr_curve', type=str, required=False, default=True, help='show the precision-recall curve')
  args = parser.parse_args()


  import warnings
  warnings.filterwarnings("ignore")


  predictions = os.listdir(args.pred_path)
  print("Number of predictions: ", len(predictions))


  all_preds = np.array([])
  all_trues = np.array([])
  all_weigh = np.array([])

  # load images
  for path in predictions:
    # load the predictions
    img_pred = cv2.imread(os.path.join(args.pred_path, path), cv2.IMREAD_COLOR)[:,:,2] / 255.0
#    img_pred = cv2.resize(img_pred, (512,512))
    from skimage.morphology import skeletonize
    img_cp = img_pred.copy()
    img = skeletonize(img_pred>0.0)
#    img_pred = img_pred * img
    img_cp_2 = img.copy()


    # load and apply mask
#    mask = cv2.imread(os.path.join(args.true_path.replace('labels', 'masks'), path.replace('.jpg', '.bmp')), cv2.IMREAD_GRAYSCALE) / 255.0    
#    img_pred = img_pred * mask

    # load the ground truth
    path = path.replace('.jpg', '.png')
    print("Path: ", path)
    img = cv2.imread(os.path.join(args.true_path, path), cv2.IMREAD_GRAYSCALE) / 255.0

    print("np.unique(img): ", np.unique(img))
    img = np.where(img>=0.1, 1.0, 0.0)

    if len(np.unique(img)) > 2:
       print(np.unique(img))
       print("Warning: Ground Truth (", path, ") is not binary, thus will be binarized at 0.3")
       all_weigh = np.append(all_weigh, np.where((0.1<=img) * (img<0.3), 0.0, 1.0).flatten(), axis=0)
#       img = np.where(img>=0.3, 1.0, 0.0)
       img = np.where(img>=0.9, 1.0, 0.0)

#    img = skeletonize(img>0.0)

    tol = 2

    if False:
      img_tmp = img_pred.copy()
      for i in range(tol, img.shape[0]-tol):
        for j in range(tol, img.shape[1]-tol):
          if img_pred[i,j] > 0:
            idxs = np.argwhere(img[i-tol:i+tol+1,j-tol:j+tol+1])
            if len(idxs) > 0:

              # update prediction
              for idx in idxs:
                if img_tmp[i-tol+idx[0],j-tol+idx[1]] < img_pred[i,j]:
                  img_tmp[i-tol+idx[0],j-tol+idx[1]] = img_pred[i,j]

              # delete prediction value
              img_pred[i,j] = 0
              img_tmp[i,j] = 0
      img_pred = img_tmp
    all_trues = np.append(all_trues, img.flatten(), axis=0)  
    all_preds = np.append(all_preds, img_pred.flatten(), axis=0)


    if True: 
      plt.subplot(221)
      plt.imshow(img_cp)
      plt.subplot(222)
      plt.imshow(img_cp_2)
      plt.subplot(223)
      plt.imshow(img)
      plt.subplot(224)
      plt.imshow(img_pred)
      mng = plt.get_current_fig_manager()
      mng.full_screen_toggle()
      plt.show()
      print(np.max(img_cp), np.max(img))
      print("AP: ", sklearn.metrics.average_precision_score(img.flatten(), img_pred.flatten()))


    
  

  # precision recall curve
  precision, recall, thresholds = sklearn.metrics.precision_recall_curve(all_trues, all_preds)

  # average precision
  ap = sklearn.metrics.average_precision_score(all_trues, all_preds, average='micro')
  print("AP: ", ap)

  # f1 score: loop over thresholds to find best f1
  f1 = []
  for thresh in thresholds[0:len(thresholds):2]:
    sys.stdout.write("\rThreshold Progress %f" % thresh)
    sys.stdout.flush()
    all_preds_tmp = all_preds.copy()
    all_preds_tmp[all_preds>=thresh] = 1.0
    all_preds_tmp[all_preds<thresh] = 0.0
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
  if args.show_pr_curve:
    # plot no skill
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.')
    # show the plot
    plt.show()

#evaluation report?
