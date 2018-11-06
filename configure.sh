#!/bin/bash
echo "Installing all necessary dependencies..."
    pip install nibabel
    pip install requests
    pip install multiprocessing
    pip install tensorflow-gpu
    pip install keras
    pip install opencv-python
    pip install matplotlib
    pip install numpy
    pip install scipy
    pip install scikit-image
    pip install scikit-learn
    pip install scikit-build
    pip install simpleitk
    pip install IPython
    pip install imgaug
    pip install tensorlayer
    pip install dipy
echo "Done with installs!"
echo "Pulling BRATS 2018 Data down, this may take a few minutes..."
if command -v python3 &>/dev/null; then
  python3 Utils/getBraTs2018Data.py
else
  python Utils/getBraTS2018Data.py
fi
echo "Success! You should be good to go! :)" 


