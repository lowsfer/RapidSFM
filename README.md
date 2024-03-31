# RapidSFM
A CUDA-accelerated high-performance implementation for the structure from motion (SFM) algorithm. Compare to most alternatives, RapidSFM performance is about two orders of magnitude faster.

## Example usage
 1. Download drone images from https://s3.amazonaws.com/DroneMapper_US/example/DroneMapper_Golf9_May2016.zip (Thank DroneMapper for sharing sample images)
 2. Extract the image files
 3. Run: `rsfm -e -d ${folder_of_images}` (if the image EXIF contains 35mm equivalent focal length) or `rsfm -f ${focal_length_in_pixels} -d ${folder_of_images}`
 4. You will find cloud_${num}.{ply/nvm/rsm}, and open *.ply with MeshLab, or *.rsm with the RapidSFM-online Windows client found at https://www.rapidsfm.com

## How to build
On ubuntu 22.04, you first need to add two PPA repos:
    `add-apt-repository -y ppa:strukturag/libde265`
    `add-apt-repository -y ppa:strukturag/libheif`
Then just install the dependency packages:
    `apt-get install libboost-dev libboost-test-dev libboost-program-options-dev libboost-fiber-dev libboost-context-dev libboost-filesystem-dev libjpeg-dev libexiv2-dev libturbojpeg0-dev libgtest-dev libeigen3-dev libopencv-dev libyaml-cpp-dev libcereal-dev rapidjson-dev libgeographic-dev cmake libheif-dev cuda`
Then just clone this repo, initialize submodules and use the normal cmake build process.

For now, It's only tested on Nvidia Turing (SM_75) and Ampere (SM_86) GPUs but it should work well on Ada (SM_89), too.
