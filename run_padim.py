import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
 
from torch.utils.data import  DataLoader
from torchvision import models
from efficientnet_pytorch import EfficientNet 
import torchvision.transforms as transforms
import torchvision.datasets as dataset

from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import cv2



ps_img = cv2.imread(r'J2901_3_UniformLight-OK226.jpg') # OK for train
ps_img = cv2.cvtColor(ps_img, cv2.COLOR_BGR2RGB)

ng_img = cv2.imread(r'J2901_3_UniformLight-NG971.jpg') # NG for test
ng_img = cv2.cvtColor(ng_img, cv2.COLOR_BGR2RGB)



def load_pkl_data(filename):
    with open(filename, "rb") as the_file:
        return pickle.load(the_file)
        the_file.close()


def save_pkl_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)



def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def produce_more_imgs(img, transform, num_img = 5):
    imgs = transform(img).unsqueeze(0)
    if num_img <= 1:
        return imgs
    else:
        for i in range(num_img-1):
            imgs = torch.cat((imgs, transform(img).unsqueeze(0)),0)
        return imgs


    
def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x
    
    
def train_by_extract_feat(backbone = 'resnet-50',
        transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=[.8,1.5], contrast=0.1, saturation=0, hue=0),
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])):
     
    
    print('start training...')
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    
        
    if backbone == 'efficientnet-b0':
        print('using efficientnet-b0...')
        model = EfficientNet.from_pretrained('efficientnet-b0')
        
        # for simplicity, to consider the following blocks
        model._blocks[0].register_forward_hook(hook)
        model._blocks[1].register_forward_hook(hook)
        model._blocks[3].register_forward_hook(hook)
        
    else:
        print('using default model resnet-50...')
        model = models.resnet50(pretrained=True)
        
        # for simplicity, to consider the following layers
        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)
    
    
    #print the model
    model.eval()
    # print(model)
  
    # image augmentation
    imgs = produce_more_imgs(ps_img,transform,25)
    print('input size: ',imgs.size())
    
    
    # use gpu to boost up computational speed
    if(torch.cuda.is_available()):
        model = model.cuda()
        imgs = imgs.cuda()
        torch.backends.cudnn.benchmark=True
    
    # pass the imgs to the model
    with torch.no_grad():
        _ = model(imgs)
    for i, layer_output in enumerate(outputs):
        print('Layer%d output size: '%(i+1),layer_output.size())
        
    
    # to embed the chosen features
    embedding_vectors = outputs[0]
    for ind, layer in enumerate(outputs):
        if ind > 0:
            embedding_vectors = embedding_concat(embedding_vectors.cpu(), layer.cpu())
    print('embedding size: ', embedding_vectors.size()) #torch.Size([5, 1792, 56, 56]) for resent-50
    
    
    
    # calculate multivariate Gaussian distribution
    ''' in this case, we uniformly extract channels with self-defined "18". 
    Instead, randomly-picking channels is implemented in original paper for resnet'''
    if backbone == 'efficientnet-b0':
        embedding_vectors = embedding_vectors[:,:,:,:]
    else:
        embedding_vectors = embedding_vectors[:,::18,:,:]
        
    print('new embedding size: ', embedding_vectors.size())
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0).numpy()
    cov = torch.zeros(C, C, H * W).numpy()
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
    
    
    # save learned distribution
    statistic_outputs = [mean, cov]
    save_pkl_data(f'weights_{backbone}.pkl', statistic_outputs)
    print('save mean and cov successfully!')
    
    
    
    # show a few layer of features
    for num_layer in range(len(outputs)):
        print("Layer ",num_layer+1)
        plt.figure(figsize=(12, 5))
        layer_viz = outputs[num_layer][0, :, :, :] # first sample
        
        if(torch.cuda.is_available()):
            layer_viz = layer_viz.cpu().data
            imgs = imgs.cpu().data
        else:
            layer_viz = layer_viz.data
        
        for i, channel_map in enumerate(layer_viz):
            if i == 5: # show 5 features
                break
            plt.subplot(1, 5, i + 1)
            plt.imshow(channel_map)
            plt.axis("off")
        plt.show()
        plt.close()
        

        
        
        
def test_by_anomaly_detection(backbone = 'resnet-50',
        weights_file = "weights_resnet-50.pkl",
        threshold = 0.5,
        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])):
    
    
    print('start testing...')
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    
        
    if backbone == 'efficientnet-b0':
        print('using efficientnet-b0...')
        model = EfficientNet.from_pretrained('efficientnet-b0')
        
        # for simplicity, to consider the following blocks
        model._blocks[0].register_forward_hook(hook)
        model._blocks[1].register_forward_hook(hook)
        model._blocks[3].register_forward_hook(hook)
        
    else:
        print('using default model resnet-50...')
        model = models.resnet50(pretrained=True)
        
        # for simplicity, to consider the following layers
        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)
    
    
    #print the model
    model.eval()
    # print(model)
    
  
    # image preprocess, do not implement augmentation here
    imgs = produce_more_imgs(ng_img,transform,1)
    print('input size: ',imgs.size())
    
    
    # use gpu to boost up computational speed
    if(torch.cuda.is_available()):
        model = model.cuda()
        imgs = imgs.cuda()
        torch.backends.cudnn.benchmark=True
    
    # pass the imgs to the model
    with torch.no_grad():
        _ = model(imgs)
    for i, layer_output in enumerate(outputs):
        print('Layer%d output size: '%(i+1),layer_output.size())
    
    
    # to embed the chosen features
    embedding_vectors = outputs[0]
    for ind, layer in enumerate(outputs):
        if ind > 0:
            embedding_vectors = embedding_concat(embedding_vectors.cpu(), layer.cpu())
    print('embedding size: ', embedding_vectors.size()) #torch.Size([5, 1792, 56, 56]) for resent-50
    
    
    # load multivariate Gaussian distribution from train
    ''' in this case, we uniformly extract channels with self-defined "18". 
    Instead, randomly-picking channels is implemented in original paper for resnet'''
    if backbone == 'efficientnet-b0':
        embedding_vectors = embedding_vectors[:,:,:,:]
    else:
        embedding_vectors = embedding_vectors[:,::18,:,:]
    
    print('new embedding size: ', embedding_vectors.size())
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
    statistic_outputs = load_pkl_data(weights_file)
    
    # calculate distance matrix
    dist_list = []
    for i in range(H * W):
        mean = statistic_outputs[0][:, i]
        conv_inv = np.linalg.inv(statistic_outputs[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)
    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
    
    # do upsampling
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=224, mode='bilinear',
                              align_corners=False).squeeze().numpy()
    score_map = score_map.reshape(-1,224,224)
    print('the size of score map: ', score_map.shape)
    
    
    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)
    
    
    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    
    # show the predicting result as heatmap and mask
    for i in range(len(scores)):
        
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        
        fig_img, ax_img = plt.subplots(1, 3, figsize=(12, 3))
        ax_img[0].imshow(imgs[i].cpu().permute(1,2,0))
        ax_img[0].title.set_text('Image')
        ax_img[0].axis('off')
        
        ax_img[1].imshow(mask, cmap='gray')
        ax_img[1].title.set_text('Predicted mask')
        ax_img[1].axis('off')
        
        ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(imgs[i].cpu().permute(1,2,0), cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[2].axis('off')
    
    
    plt.tight_layout()
    plt.show()
    
    
    
        
if __name__ == "__main__":
    # for resent-50
    train_by_extract_feat()
    test_by_anomaly_detection(threshold=0.7)
    # for efficientnet-b0
    train_by_extract_feat(backbone = 'efficientnet-b0')
    test_by_anomaly_detection(backbone = 'efficientnet-b0', threshold=0.7, weights_file = 'weights_efficientnet-b0.pkl')
    
    