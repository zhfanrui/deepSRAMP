mkdir blast && cd blast

wget -c https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.16.0+-x64-linux.tar.gz 
wget -c https://github.com/rcedgar/muscle/releases/download/v5.2/muscle-linux-x86.v5.2

tar zxvf ncbi-blast-2.16.0+-x64-linux.tar.gz 
mv ncbi-blast-2.16.0+/bin ./bin
mv muscle-linux-x86.v5.2 bin/muscle
chmod +x bin/muscle
rm -rf ncbi-blast-2.16.0+/

wget -c https://github.com/zhfanrui/deepSRAMP/releases/download/Database/hg38_mature.tgz
wget -c https://github.com/zhfanrui/deepSRAMP/releases/download/Database/mm39_mature.tgz
tar zxvf hg38_mature.tgz
tar zxvf mm39_mature.tgz
