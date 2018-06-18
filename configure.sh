#!/bin/bash
echo "Downgrading pip install to version 9 for imports and installs. Don't worry, it'll be upgraded once this is finished"
if command -v python3 &>/dev/null; then
  python3 -m pip install --upgrade pip==9.0.3
  python3 Utils/setup.py
  echo "Done with installs! Pulling BRATS 2018 Data down, this may take a few minutes..."
  python3 Utils/getBraTs2018Data.py
  echo "Done! Upgrading pip back to version 10..."
  python3 -m pip install --upgrade pip
else
  python -m pip install --upgrade pip==9.0.3
  python Utils/setup.py
  echo "Done with installs! Pulling BRATS 2018 Data down, this may take a few minutes..."
  python Utils/getBraTS2018Data.py
  echo "Upgrading pip back to version 10..."
  python -m pip install --upgrade pip
fi
echo "Now to pull down the Mask R-CNN submodule..." 
git submodule update --init --recursive --remote
echo "Success! You should be good to go! :)" 


