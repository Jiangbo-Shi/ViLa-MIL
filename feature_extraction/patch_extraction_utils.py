### Dependencies
# Base Dependencies
import os
# LinAlg / Stats / Plotting Dependencies
from concurrent.futures import ThreadPoolExecutor
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# Torch Dependencies
import torch
import torch.multiprocessing
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import transforms
device = torch.device('cuda')
torch.multiprocessing.set_sharing_strategy('file_system')

# Model Architectures
from nn_encoder_arch.vision_transformer import vit_small
from nn_encoder_arch.resnet_trunc import resnet50_trunc_baseline


def eval_transforms_clip(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    trnsfrms_val = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Resize((224, 224)),
                                       transforms.Normalize(mean = mean, std = std)])
    return trnsfrms_val

def eval_transforms(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5,0.5,0.5), (0.5,0.5,0.5)
    trnsfrms_val = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Normalize(mean = mean, std = std)])
    return trnsfrms_val


def torchvision_ssl_encoder(name: str, pretrained: bool = False, return_all_feature_maps: bool = False):
    pretrained_model = getattr(resnets, name)(pretrained=pretrained, return_all_feature_maps=return_all_feature_maps)
    pretrained_model.fc = Identity()
    return pretrained_model


def save_embeddings(model, fname, dataloader, enc_name, overwrite=False):

    if os.path.isfile('%s.h5' % fname) and (overwrite == False):
        return None

    embeddings, coords, file_names = [], [], []

    for batch, coord in dataloader:
        with torch.no_grad():
            batch = batch.to(device)
            if(enc_name != 'clip_RN50' and enc_name != 'clip_ViTB32'):
                embeddings.append(model(batch).detach().cpu().numpy().squeeze())
            else:
                embeddings.append(model.encode_image(batch).detach().cpu().numpy().squeeze())
            file_names.append(coord)

    for file_name in file_names:
        for coord in file_name:
            coord = coord.rstrip('.png').split('_')
            coords.append([int(coord[0]), int(coord[1])])

    print(fname)

    embeddings = np.vstack(embeddings)
    coords = np.vstack(coords)

    f = h5py.File(fname+'.h5', 'w')
    f['features'] = embeddings
    f['coords'] = coords
    f.close()


def create_embeddings(embeddings_dir, enc_name, dataset, batch_size, save_patches=False,
                      patch_datasets='path/to/patch/datasets', assets_dir ='./ckpts/',
                      disentangle=-1, stage=-1):
    print("Extracting Features for '%s' via '%s'" % (dataset, enc_name))
    if enc_name == 'resnet50_trunc':
        model = resnet50_trunc_baseline(pretrained=True)
        eval_t = eval_transforms(pretrained=True)
    
    elif enc_name == 'clip_RN50':
        import clip 
        model, preprocess = clip.load("RN50", device=device)
        eval_t = eval_transforms_clip(pretrained=True)
    
    elif enc_name == 'clip_ViTB32':
        import clip 
        model, preprocess = clip.load("ViT-B/32", device=device)
        eval_t = eval_transforms_clip(pretrained=True)

    elif enc_name == 'model_dino' or enc_name == 'dino_HIPT':
        ckpt_path = os.path.join(assets_dir, enc_name+'.pth')
        assert os.path.isfile(ckpt_path)
        model = vit_small(patch_size=16)
        state_dict = torch.load(ckpt_path, map_location="cpu")['teacher']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        #print("Missing Keys:", missing_keys)
        #print("Unexpected Keys:", unexpected_keys)
        eval_t = eval_transforms(pretrained=False)

    elif enc_name == 'model_simclr':
        ckpt_path = os.path.join(assets_dir, enc_name+'.pt')
        assert os.path.isfile(ckpt_path)
        model = torchvision_ssl_encoder('resnet50', pretrained=True)
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(ckpt_path), strict=False)
        eval_t = eval_transforms(pretrained=False)

    elif enc_name == 'model_simclr_histo_res18':
        ckpt_path = os.path.join(assets_dir, enc_name+'.ckpt')
        assert os.path.isfile(ckpt_path)

        model = torchvision.models.__dict__['resnet18'](pretrained=False)

        state = torch.load(ckpt_path, map_location='cpu')
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    else:
        pass

    model = model.to(device)
    if(enc_name != 'clip_ViTB32' and enc_name != 'clip_RN50'):
        model = torch.nn.DataParallel(model)
    model.eval()

    if 'dino' in enc_name:
        _model = model
        if stage == -1:
            model = _model
        else:
            model = lambda x: torch.cat([x[:, 0] for x in _model.get_intermediate_layers(x, stage)], dim=-1)

    if stage != -1:
        _stage = '_s%d' % stage
    else:
        _stage = ''

    if dataset == 'TCGA':
        # pool = ThreadPoolExecutor(max_workers=48)
        for wsi_name in tqdm(os.listdir(patch_datasets)):
            dataset = PatchesDataset(os.path.join(patch_datasets, wsi_name), transform=eval_t)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            fname = os.path.join(embeddings_dir, wsi_name)
            if(not os.path.exists(fname)):
                save_embeddings(model, fname, dataloader, enc_name)

                # args = [model, fname, dataloader]
                # pool.submit(lambda p: save_embeddings(*p), args)
        # pool.shutdown(wait=True)


class PatchesDataset(Dataset):
    def __init__(self, file_path, transform=None):
        file_names = os.listdir(file_path)
        imgs = []
        coords = []
        for file_name in file_names:
            imgs.append(os.path.join(file_path, file_name))
            coords.append(file_name)
        self.imgs = imgs
        self.coords = coords
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        coord = self.coords[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, coord

    def __len__(self):
        return len(self.imgs)
        
