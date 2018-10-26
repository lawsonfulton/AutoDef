### install Anaconda
mkdir deps/
wget -O deps/anaconda.sh https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh
bash deps/anaconda.sh -b -p ./extern/anaconda

# Other Deps
sudo apt-get install build-essential curl git cmake unzip autoconf autogen automake libtool mlocate zlib1g-dev g++-7 python python3-numpy python3-dev python3-pip python3-wheel wget libboost-all-dev pkg-config zip g++ zlib1g-dev unzip python
sudo updatedb



### For tensorflow cc

#Bazel
wget -O deps/bazel.sh https://github.com/bazelbuild/bazel/releases/download/0.18.0/bazel-0.18.0-installer-linux-x86_64.sh
bash deps/bazel.sh --user

cd extern/tensorflow_cc/tensorflow_cc
mkdir build && cd build
# for static library only:
cmake ..
make -j8 && sudo make install


#TODO Make sure that I can run tensorflow from tensorflowcc
extern/anaconda/bin/pip install keras==2.0.8 #tensorflow==1.8.0
