#!/bin/bash
echo "Pulling BRATS 2018 Data down, this may take a few minutes..."
if command -v python3 &>/dev/null; then
  python3 Utils/getBraTs2018Data.py
else
  python Utils/getBraTS2018Data.py
fi
echo "Data successfully pulled down! Now to pull down the Mask R-CNN submodule..." 
git submodule update --init --recursive --remote
echo "Success! You should be good to go! :)" 
