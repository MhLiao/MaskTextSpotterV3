wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1604-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

nvidia-smi
echo "export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}" >> ~/.profile
source ~/.profile

nvcc --version
gcc --version
python3 --version

wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
export PATH=~/anaconda3/bin:$PATH
conda --version
conda update -n base -c defaults conda

conda create --name masktextspotter -y
  conda activate masktextspotter

  # this installs the right pip and dependencies for the fresh python
  conda install ipython pip

  # python dependencies
  pip install ninja yacs cython matplotlib tqdm opencv-python shapely scipy tensorboardX pyclipper Polygon3 editdistance 

  # install PyTorch


conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch


  export INSTALL_DIR=$PWD

  # install pycocotools
  cd $INSTALL_DIR
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  python setup.py build_ext install

  # install apex
  cd $INSTALL_DIR
  git clone https://github.com/NVIDIA/apex.git
  cd apex
  python setup.py install --cuda_ext --cpp_ext

  # clone repo
  cd $INSTALL_DIR
  git clone https://github.com/Nikeliza/MaskTextSpotterV3.git
  cd MaskTextSpotterV3

  # build
  python setup.py build develop


  unset INSTALL_DIR

pip install requests
sudo apt install unzip
mkdir output
mkdir output/mixtrain
mkdir ~/MaskTextSpotterV3/datasets

cd ~
git clone https://github.com/chentinghao/download_google_drive.git
cd download_google_drive
python download_gdrive.py 1XQsikiNY7ILgZvmvOeUf9oPDG4fTp0zs ~/MaskTextSpotterV3/output/mixtrain/trained_model.pth

cd ~/download_google_drive
python download_gdrive.py 1sptDnAomQHFVZbjvnWt2uBvyeJ-gEl-A ~/MaskTextSpotterV3/datasets/icdar2013.zip
cd ~/MaskTextSpotterV3/datasets
unzip icdar2013.zip

cd ~/download_google_drive
python download_gdrive.py 1HZ4Pbx6TM9cXO3gDyV04A4Gn9fTf2b5X ~/MaskTextSpotterV3/datasets/icdar2015.zip
cd ~/MaskTextSpotterV3/datasets
unzip icdar2015.zip

wget http://www.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip
unzip SynthText.zip
mv SynthText train_images
mkdir synthtext
mv train_images synthtext

cd ~/download_google_drive
python download_gdrive.py 1pCmL5iZuVpLI1q32xvQMj-4b6R8znlkE ~/MaskTextSpotterV3/datasets/synthtext/label.tar.gz
cd ~/MaskTextSpotterV3/datasets/synthtext
tar xvzf label.tar.gz
mv SynthText_GT_E2E train_gt
find ~/MaskTextSpotterV3/datasets/synthtext/train_images -type f >> ../train_list.txt

cd ~/download_google_drive
python download_gdrive.py 1BpE2GEFF7Ay7jPqgaeHxMmlXvM-1Es5_ ~/MaskTextSpotterV3/datasets/scut.zip
cd ~/MaskTextSpotterV3/datasets
unzip scut.zip

cd ~/download_google_drive
python download_gdrive.py 1bC68CzsSVTusZVvOkk7imSZSbgD1MqK2 ~/MaskTextSpotterV3/datasets/total.zip
cd ~/MaskTextSpotterV3/datasets
unzip total.zip
mv Images total_text

cd ~/download_google_drive
python download_gdrive.py 1KevcLcmZr0FQVBfplzCcriaqdpWSF3K2 ~/MaskTextSpotterV3/datasets/total_text/label.zip
cd ~/MaskTextSpotterV3/datasets/total_text
unzip label.zip
mv Train train_images
mv Test test_images
mv total_text_labels/test_gts .
mv total_text_labels/train_gts .
#переименовать!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


sudo add-apt-repository ppa:nilarimogard/webupd8
sudo apt-get update
sudo apt-get install grive

mkdir ~/Google-Drive
cd ~/Google-Drive
grive -a --id 480936158995-g5qcivj1df5qv3ccq5115o4g71nrgil7.apps.googleusercontent.com --secret 1whdLc3Hpn0rKB_hnoD34mud -s diplom_drive
#mkdir ~/MaskTextSpotterV3/datasets/synthtext


#nano configs/mixtrain/seg_rec_poly_fuse_feature.yaml

#cd ~/MaskTextSpotterV3
#sh test.sh

#/home/nikiforova_er/MaskTextSpotterV3/file.tar.gz
