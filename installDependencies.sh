
### install Anaconda
CONTREPO=https://repo.continuum.io/archive/
# Stepwise filtering of the html at $CONTREPO
# Get the topmost line that matches our requirements, extract the file name.
ANACONDAURL=$(wget -q -O - $CONTREPO index.html | grep "Anaconda3-" | grep "Linux" | grep "86_64" | head -n 1 | cut -d \" -f 2)
mkdir deps/
wget -O deps/anaconda.sh $CONTREPO$ANACONDAURL
bash deps/anaconda.sh


# Keras stuff? Libigl python bindinggsss



### For tensorflow cc
sudo apt-get install build-essential curl git cmake unzip autoconf autogen automake libtool mlocate \
                     zlib1g-dev g++-7 python python3-numpy python3-dev python3-pip python3-wheel wget
sudo updatedb

cd extern/tensorflow_cc/tensorflow_cc
mkdir build && cd build
# for static library only:
cmake ..
make -j8 && sudo make install

