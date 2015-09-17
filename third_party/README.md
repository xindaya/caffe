# third_party
Common C++ dependencies for Petuum Bosen. You will need to download and build before compiling Bosen.

To download and build, run the following commands from the root directory of your Petuum Bosen repository:

```
sudo apt-get -y install g++ make python-dev libxml2-dev libxslt-dev git zlibc zlib1g zlib1g-dev libbz2-1.0 libbz2-dev
git clone https://github.com/petuum/third_party.git
cd third_party
make -j2
```
