### install Anaconda
mkdir deps/
wget -O deps/anaconda.sh https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh
bash deps/anaconda.sh -b -p ./extern/anaconda

# Other Deps
sudo apt-get install build-essential curl git cmake unzip autoconf autogen automake libtool mlocate zlib1g-dev g++-7 python python3-numpy python3-dev python3-pip python3-wheel wget libboost-all-dev pkg-config zip g++ zlib1g-dev unzip python -y
sudo updatedb



### For tensorflow cc
# Bazel
wget -O deps/bazel.sh https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-installer-linux-x86_64.sh
sudo bash deps/bazel.sh

# build and install my version of tensorflow
cd extern/tensorflow_cc/tensorflow_cc
mkdir build
cd build
cmake -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED=ON ..
make -j8 && sudo make install
cd tensorflow
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./pip_package_build
cd ../../../../../ # back to AutoDef/
extern/anaconda/bin/pip install --upgrade extern/tensorflow_cc/tensorflow_cc/build/tensorflow/pip_package_build/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl

# Install Keras
extern/anaconda/bin/pip install keras==2.0.8 

# Now I need to set the default in keras
mkdir ~/.keras/
cp ./keras.json ~/.keras/


### Now the rest of the related code
# Build GAUSS
cd extern/GAUSS
bash InstallGAUSS_Ubuntu_noqt.sh
cd ../../

# Build Libigl Python bindings
cd extern/libigl/python
mkdir build
cd build; cmake ..; make -j8;
cd ../../../../

# Build Cubacode
cd src/cubacode
mkdir build
cd build
cmake ..
make -j8
cd ../../../
