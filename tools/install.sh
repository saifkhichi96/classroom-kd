#!/bin/bash
chmod +x .
  
# put your install commands here:
apt update
apt clean
pip install efficientnet_pytorch 
pip install seaborn

"$@"

# pip uninstall torchtext
# pip install --upgrade torch torchvision
# # conda install pytorch==1.10.1 torchvision==0.15.1 cudatoolkit=11.3 -c pytorch -y
# # pip uninstall torchvision torchtext pytorch-lightning lightning
# # pip install torch==1.9
# # pip install torchvision torchtext pytorch-lightning lightning
# pip install efficientnet_pytorch