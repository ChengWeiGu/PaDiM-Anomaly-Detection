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

from os.path import join, basename, exists
from os import mkdir, walk, listdir
import datetime
import random


# In this case, we use a OK image to produce lots of augmented images
ps_img = cv2.imread(r'J2901_3_UniformLight-OK226.jpg')
ps_img = cv2.cvtColor(ps_img, cv2.COLOR_BGR2RGB)



def load_pkl_data(filename):
    with open(filename, "rb") as the_file:
        return pickle.load(the_file)
        the_file.close()


def save_pkl_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def shuffle_channels(num_channels):
    indices = list(range(num_channels))
    random.shuffle(indices)
    return indices


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


def augment_imgs(img, num_img = 10, 
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=[.8,1.5], contrast=0.1, saturation=0, hue=0),
        transforms.ToTensor()
    ])):
    
    for itr in range(num_img):
        _img = transform(img).permute(1,2,0).numpy()*255
        _img = cv2.cvtColor(_img.astype('uint8'), cv2.COLOR_RGB2BGR)
        filename = join('./imgs/ps/{}_OK_{}.jpg'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),itr))
        cv2.imwrite(filename,_img)
        print('save images to {}'.format(filename))

    
def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x
    

def import_images(path,transform):
    imgs = []
    count = 0
    for root, dirs, files in walk(path):
        for f in files:
            if f.endswith('.jpg'):
                count+=1
                filename = join(root,f)
                _img = cv2.imread(filename)
                _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
                _img = transform(_img).unsqueeze(0)
                if count == 1:
                    imgs = _img
                else:
                    imgs = torch.cat((imgs,_img),0)
    
    return imgs
    


def main(backbone = 'resnet-50',
        threshold = 0.5,
        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])):
    
    
    def hook(module, input, output):
        outputs.append(output)  
    
    
    def calculate_embedding(model,imgs):
        
        if(torch.cuda.is_available()):
            model = model.cuda()
            imgs = imgs.cuda()
            torch.backends.cudnn.benchmark=True
        
        
        with torch.no_grad():
            _ = model(imgs)
        for i, layer_output in enumerate(outputs):
            print('Layer%d output size: '%(i+1),layer_output.size())
            
        
        # to embed the chosen features
        embedding_vectors = outputs[0]
        for ind, layer in enumerate(outputs):
            if ind > 0:
                embedding_vectors = embedding_concat(embedding_vectors.cpu(), layer.cpu())
        print('embedding size: ', embedding_vectors.size())
        
        return embedding_vectors
        
        
    def calculate_score_map(statistic_outputs, embedding_vectors):
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
    
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
            
        # to ensure that size is 3-dim (B,H,W) if only one image 
        score_map = score_map.reshape(-1,224,224)
        
        return score_map, score_map.max(), score_map.min()
    
    
    
    print('start training...')
        
    if backbone == 'efficientnet-b0':
        print('using efficientnet-b0...')
        model = EfficientNet.from_pretrained('efficientnet-b0')
        
        # for simplicity, to consider the following blocks
        model._blocks[0].register_forward_hook(hook)
        model._blocks[1].register_forward_hook(hook)
        model._blocks[3].register_forward_hook(hook)
        
    elif backbone == 'efficientnet-b5':
        print('using efficientnet-b5...')
        model = EfficientNet.from_pretrained('efficientnet-b5')
        
        # for simplicity, to consider the following blocks
        model._blocks[0].register_forward_hook(hook)
        model._blocks[5].register_forward_hook(hook)
        model._blocks[8].register_forward_hook(hook)
        
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
    
    
    # read images as tensor
    train_imgs = import_images('./imgs/ps',transform=transform)
    print('input size of train_imgs: ',train_imgs.size())
    
    
    # initialize hook outputs for training
    outputs = []
    embedding_vectors = calculate_embedding(model,train_imgs)
    
    
    # calculate multivariate Gaussian distribution
    ''' in this case, we randomly extract channels with suffle function'''    
    if backbone == 'efficientnet-b0':
        num_channels = embedding_vectors.size(1) # torch.Size([20, 80, 112, 112]) for b0
        indices = shuffle_channels(num_channels)[:60]
        
    elif backbone == 'efficientnet-b5':
        num_channels = embedding_vectors.size(1) # torch.Size([20, 128, 112, 112]) for b5
        indices = shuffle_channels(num_channels)[:60]
        
    else:
        num_channels = embedding_vectors.size(1) # torch.Size([20, 1792, 56, 56]) for resent-50
        indices = shuffle_channels(num_channels)[:100]
    
    embedding_vectors = embedding_vectors[:,indices,:,:]
    print('new embedding size: ', embedding_vectors.size())
    
    
    # calculate mean and cov
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0).numpy()
    cov = torch.zeros(C, C, H * W).numpy()
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
    
    
    # save learned distribution
    statistic_outputs = [mean, cov]
    weights_file = f'weights_{backbone}.pkl'
    save_pkl_data(weights_file, statistic_outputs)
    print('save mean and cov successfully!\n\n')
    
    
    # calculate distance matrix and score map
    score_map_train, max_train_score, min_train_score = calculate_score_map(statistic_outputs, embedding_vectors.numpy())
    
    ''' show a few layer of features
    for num_layer in range(len(outputs)):
        print("Layer ",num_layer+1)
        plt.figure(figsize=(12, 5))
        layer_viz = outputs[num_layer][0, :, :, :] # first sample
        
        if(torch.cuda.is_available()):
            layer_viz = layer_viz.cpu().data
            train_imgs = train_imgs.cpu().data
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
    '''
    
    print('start testing...')
    
    # the inference image as tensor
    test_imgs = import_images('./imgs/test',transform=transform)
    print('input size of test_imgs: ',test_imgs.size())
    
    
    # initialize hook outputs for testing
    outputs = []
    embedding_vectors = calculate_embedding(model,test_imgs)
    embedding_vectors = embedding_vectors[:,indices,:,:]
    print('new embedding size: ', embedding_vectors.size())
    
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    # statistic_outputs = load_pkl_data(weights_file)
    
    # calculate distance matrix and score map
    score_map_test, max_test_score, min_test_score = calculate_score_map(statistic_outputs, embedding_vectors.numpy())
    
    
    # Normalization
    max_score = max_test_score if max_test_score > max_train_score else max_train_score
    min_score = min_train_score if min_train_score < min_test_score else min_test_score
    print('max score: ',max_score)
    print('min score: ', min_score)
    
    scores = (score_map_test - min_score) / (max_score - min_score)
    
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    
    # show the predicting result as heatmap and mask
    for i in range(len(scores)):
        
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        
        im = denormalization(test_imgs[i].cpu().numpy())
        
        fig_img, ax_img = plt.subplots(1, 3, figsize=(12, 3))
        ax_img[0].imshow(im)
        ax_img[0].title.set_text('Image')
        ax_img[0].axis('off')
        
        ax_img[1].imshow(mask, cmap='gray')
        ax_img[1].title.set_text('Predicted mask')
        ax_img[1].axis('off')
        
        ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(im, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[2].axis('off')
    
    
    plt.tight_layout()
    plt.show()
    
    
    

if __name__ == "__main__":
    
    # image augmentation if needed
    augment_imgs(ps_img, num_img=10)
    
    # run and go through all models
    for backbone in ['resnet-50','efficientnet-b0','efficientnet-b5']:
        main(backbone = backbone, threshold = 0.7)
        
    # run a given model
    main(backbone = 'efficientnet-b5', threshold = 0.7)
    