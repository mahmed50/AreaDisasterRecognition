#!/bin/bash

# Download the datasets
mkdir ader/Data
cd ader/Data
mkdir -p AIDER
cd AIDER
echo "Downlaoding AIDERV2 Test"
wget https://zenodo.org/records/10891054/files/Test.zip
echo "Downloading AIDERV2 Train"
wget https://zenodo.org/records/10891054/files/Train.zip
echo "Downloading AIDERV2 Val"
wget https://zenodo.org/records/10891054/files/Validation.zip
echo "Unzip AIDERV2 Test"
unzip Test.zip
echo "Unzip AIDERV2 Train"
unzip Train.zip
echo "Unzip AIDERV2 Val"
unzip Validation.zip
mv Validation Val
cd ..
cd ..
cd src
