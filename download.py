import requests
from tqdm import tqdm
import os
import shutil
from urllib.parse import urlparse, unquote
import kaggle

def download_unpack(path, url):
    os.makedirs(path, exist_ok=True)
    filename = os.path.basename(unquote(urlparse(url).path))
    full_zip_path = f'{path}/{filename}'
    full_path = os.path.splitext(full_zip_path)[0]
    if os.path.isdir(full_path):
        print(f'{full_path} already exists')
    else:
        print(f'Downloading {full_zip_path}')
        with requests.get(url, stream=True) as req:
            length = int(req.headers.get('Content-Length'))
            with tqdm.wrapattr(req.raw, 'read', total=length, desc='') as data:
                with open(full_zip_path, 'wb') as output:
                    shutil.copyfileobj(data, output)
        print(f'Unpacking {full_zip_path}')
        shutil.unpack_archive(full_zip_path, os.path.splitext(full_path)[0])
        os.remove(full_zip_path)
        print('Done')

kaggle.api.authenticate()
kaggle.api.dataset_download_files('xhlulu/flickrfaceshq-dataset-nvidia-resized-256px', path='./datasets/ffhq256', unzip=True, quiet=False)

download_unpack('./pretrained/autoencoders', 'https://ommer-lab.com/files/latent-diffusion/kl-f4.zip')