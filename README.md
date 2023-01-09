# LatentGAN

## Przygotowanie środowiska

```
git clone https://github.com/Antollo/LatentGAN
cd LatentGAN
```

Przygotowanie środowiska:

```sh
git clone https://github.com/NVlabs/stylegan3
conda env create --name LatentGAN --file ./stylegan3/environment.yml
conda activate LatentGAN
conda install torchvision -c pytorch
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/CompVis/latent-diffusion.git@main#egg=latent-diffusion
pip install kaggle omegaconf einops pytorch_lightning psutil 
```

Stylegan wymaga pełnej instalacji CUDA (razem z `nvcc`, bo kompiluje sobie własne operatory).
Pobieramy tą wersję CUDA Toolkit, która jest w środowisku condy (być może zadziała z inną).


[https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=WSLUbuntu&target_version=20&target_type=runfilelocal](https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=WSLUbuntu&target_version=20&target_type=runfilelocal)

```sh
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
sudo sh cuda_11.1.0_455.23.05_linux.run --override
```

Przydatna uwaga na później:
> To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.1/bin


Niestety ta wersja CUDA Toolkit nie działa z g++-11 (ale działa z g++-10). Być może trzeba też będzie zainstalować `build-essential`, jeśli nie ma.

```
sudo apt install gcc-10 g++-10

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
```

## Przygotowanie zbioru danych

Do łatwego pobrania zbioru danych jest potrzebny Kaggle API Token [https://www.kaggle.com/USERNAME/account](https://www.kaggle.com/USERNAME/account). Wygenerowany tam `kaggle.json` wygląda tak:

```json
{"username":"USERNAME","key":"KEY"}
```

Należy go umieścić w `$HOME/.kaggle`

```sh
# Przykładowo
echo '{"username":"antoninowinowski","key":"1a2b3c4d..."}' > $HOME/.kaggle/kaggle.json
```


Pobranie zbiorów danych i przetrenowanego enkodera:

```sh
# oryginalny zbiór danych oraz enkoder i dekoder
python download.py
# zbiór danych już przetworzony do przestrzeni latent
wget -O ./datasets/kl-latents.zip 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBakxRVGlIdDViTDNxRThsOThvUWt4bThFOFl5P2U9RThlNmhC/root/content'
```

## Uczenie sieci

Oficjalne wskazówki trenowania stylegan3 znajdują się na stronie [https://github.com/NVlabs/stylegan3/blob/main/docs/configs.md](https://github.com/NVlabs/stylegan3/blob/main/docs/configs.md).

### Oszacowanie ile należy przydzielić CPU i RAMu

Według dokumentacji stylegan2-ada (repo stylegan3 bazuje na tym) [https://github.com/NVlabs/stylegan2-ada-pytorch#expected-training-time](https://github.com/NVlabs/stylegan2-ada-pytorch#expected-training-time) należy przydzielić 31.9 GB RAM dla 8 GPU i obrazków 128x128 (obrazki w `kl-latent` są 64x64 więc to zawyżone oszacowanie), a dla obrazków 256x256 i 8 GPU 34.7 GB. StyleGAN3 w używanej przez nas konfiguracji jest ok. 1.3 razy "większy" od StyleGAN2 więc z grubsza można oszacować, że będzie potrzebne 45 GB RAMu. Na naszej maszynie obciążenie 16 rdzeniowego CPU to ok 15-20% przy jednym GPU, więc dla 8 GPU można oszacować, że potrzebne będzie ok. 20 - 25 rdzeni.

### Trening oryginalnego StyleGAN2 na ffhq256

Uruchomienie na 8 GPU. Powinno zająć ok. 37 h. Snapshoty zajmą ok. 8 GB przestrzeni dyskowej (20 x ok. 400 MB).

Jeśli okaże się, że jest za mało pamięci na GPU należy dodać `--batch-gpu=16` (lub ewentualnie mniej).

```sh
python ./stylegan3/dataset_tool.py --source=./datasets/ffhq256 --dest=./datasets/ffhq256.zip

python ./stylegan3/train.py --outdir=./training-runs --cfg=stylegan3-t --data=./datasets/ffhq256.zip --gpus=8 --batch=32 --snap=125 --gamma=2 --mirror=1 --aug=noaug --kimg 10000
```

### Trening StyleGAN3 na obrazkach w latent space

Uruchomienie na 8 GPU. Powinno zająć ok. 32 h. Snapshoty zajmą ok. 8 GB przestrzeni dyskowej (20 x ok. 400 MB).

```sh
# kl-latent.zip jest już pobrany, 
# gdyby go nie było, należałoby wywołać:
# python transform_dataset.py
# python ./stylegan3/dataset_tool.py --source=./datasets/kl-latent --dest=./datasets/kl-latent.zip

python ./stylegan3/train.py --outdir=./training-runs --cfg=stylegan3-t --data=./datasets/kl-latent.zip --gpus=8 --batch=32 --snap=125 --gamma=0.125 --mirror=0 --aug=noaug --kimg 10000
```