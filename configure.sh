#!/bin/bash

python3 Utils/getBraTs2018Data.py

git submodule update --init --recursive --remote
