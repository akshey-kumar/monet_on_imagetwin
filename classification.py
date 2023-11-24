import torch
from torch import nn
import cv2
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from models.monet import MultiScaleModel
from dataloader import MonetDL

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


parser = argparse.ArgumentParser(description='Run trained Monet model on image twin data.')
parser.add_argument('--model-path', help='path to trained model', default='./model_weights/monet_flexible_margin_model.pt', required=False)
parser.add_argument('--patch-map-root', help='path to patch mapping weights', default='./model_weights/', required=False)
parser.add_argument('--output-dir', help='path to output directory', default='./dataset/', required=False)
parser.add_argument('--image-dir', help='path to image/dataset directory', default='./dataset/', required=False)
parser.add_argument('--dataset', help='train/test/validation set', default='test', required=True)
parser.add_argument('--start-k', help='starting kernel size', default=11, required=False)
parser.add_argument('--batch-size', help='dataloader batch size', default=16, required=False)
parser.add_argument('--device', help='gpu/cpu device', default='cpu', required=False)

if __name__ == '__main__':
    
    args = parser.parse_args()

    # Load model
    model = MultiScaleModel(start_k=args.start_k,
                            patch_map_root=args.patch_map_root)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    
    model.to(args.device)
    model.eval()

    # Load dataloader on imagetwin data
    data_dl = MonetDL(dataset=args.dataset)
    args.batch_size = int(args.batch_size) 
    data_dset = DataLoader(data_dl, batch_size=args.batch_size, shuffle=False, num_workers=8)

    labels_true = np.loadtxt('dataset/data_'+ args.dataset + '/labels_'+ args.dataset).astype(int) 

    labels_pred = np.zeros_like(labels_true, dtype = int)
    # Inference loop
    with torch.no_grad():
        print('Predicting masks for samples ...')
        for img1, img2, s1, s2, doi, panel1, panel2 in tqdm(data_dset):
        
            img1 = img1.to(args.device)
            img2 = img2.to(args.device)


            sm1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(True, 
                                                                     False, 
                                                                     img1, img2, 
                                                                     None, None, None, None, None)
            sm2, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(True, 
                                                                     False, 
                                                                     img2, img1, 
                                                                     None, None, None, None, None)
        
            sm1 = sm1.to('cpu').numpy()
            sm2 = sm2.to('cpu').numpy()
            

            samples = img1.shape[0]
            for i in range(samples):

                w1, w2 = int(s1[i][1].item()), int(s2[i][1].item())  # Extract the integer value from the tensor
                h1, h2 = int(s1[i][0].item()), int(s2[i][0].item())  # Extract the integer value from the tensor

                mask1 = cv2.resize(sm1[i], (w1, h1))
                mask2 = cv2.resize(sm2[i], (w2, h2))

                mask1[mask1>0.] = 1
                mask1[mask1<=0.] = 0
                mask2[mask2>0.] = 1
                mask2[mask2<=0.] = 0
                
                print(doi[i])
                if labels_true[int(doi[i])]:
                    print('duplicate')

                frac1, frac2 = np.sum(mask1)/mask1.size, np.sum(mask2)/mask2.size  
                print(frac1, frac2)

                ### Classification part
                if frac1 >= 0.1 and frac2 >= 0.1:
                    labels_pred[int(doi[i])] = 1
                else:
                    labels_pred[int(doi[i])] = 0

                ### Saving the masks
                mask_dir = args.output_dir + 'data_' + args.dataset + '/features_' + args.dataset + '/'+ str(doi[i]) 

                os.makedirs(mask_dir, exist_ok=True)

                ### Saving the masks
                cv2.imwrite(os.path.join(mask_dir, 'mask1.png'), mask1 * 255)
                cv2.imwrite(os.path.join(mask_dir, 'mask2.png'), mask2 * 255)

    np.save('labels_pred_' + args.dataset, labels_pred)

    ### Evaluation of predicted labels
    acc = accuracy_score(labels_true, labels_pred)
    prec = precision_score(labels_true, labels_pred)
    recall = recall_score(labels_true, labels_pred)
    f1 = f1_score(labels_true, labels_pred)

    tn, fp, fn, tp = confusion_matrix(labels_true, labels_pred).ravel()

    # Create a formatted output
    output = f"   f1    rec    pre    acc  dataset\n"
    output += f"-----  -----  -----  -----  ---------\n"
    output += f"{f1:.3f}  {recall:.3f}  {prec:.3f}  {acc:.3f}  {args.dataset:<9}\n"
    print(output)



