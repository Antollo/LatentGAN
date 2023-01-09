import torch
import torchvision
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import matplotlib.pyplot as plt


def load_model_from_config(config_path, ckpt_path):
    print(f'Loading model from {ckpt_path}')
    model = instantiate_from_config(OmegaConf.load(config_path).model)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
    model.cuda()
    model.eval()
    return model

model_kl = load_model_from_config(
    './src/latent-diffusion/models/first_stage_models/kl-f4/config.yaml', 
    './pretrained/autoencoders/kl-f4/model.ckpt'
)

dataset = torchvision.datasets.ImageFolder(
    root='./datasets/ffhq256/',
    transform=torchvision.transforms.ToTensor()
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

model = model_kl

def encode(model, image):
    result = model.encode(image)
    if type(result) is tuple:
        result = result[0]
    else:
        result = result.mean
    return result.detach()

min = torch.ones(3, dtype=torch.float32).cuda() * 1000
max = torch.ones(3, dtype=torch.float32).cuda() * -1000

for i, data in tqdm(enumerate(loader), total=len(loader)):
    image, labels = data
    L = encode(model, image.cuda())
    min = torch.minimum(min, L.amin(dim=(0, 2, 3)))
    max = torch.maximum(max, L.amax(dim=(0, 2, 3)))

print(min)
print(max)
torch.save(min, './datasets/kl-latent.min.pt')
torch.save(max, './datasets/kl-latent.max.pt')
#min = torch.load('./datasets/kl-latent.min.pt')
#max = torch.load('./datasets/kl-latent.max.pt')

diff = (max - min)[:, None, None]
min = min[:, None, None]

for i, data in tqdm(enumerate(loader), total=len(loader)):
    image, labels = data

    L = encode(model, image.cuda())
    standardized = (L - min) / diff
    standardized = torch.clip(standardized, 0, 1)
    torchvision.utils.save_image(standardized, f'./datasets/kl-latent/{i}_1.png')

    image = torch.flip(image, (3,))

    L = encode(model, image.cuda())
    standardized = (L - min) / diff
    standardized = torch.clip(standardized, 0, 1)
    torchvision.utils.save_image(standardized, f'./datasets/kl-latent/{i}_2.png')