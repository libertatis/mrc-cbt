# Download SQuAD
# PWD=$(pwd)
# SQUAD_DIR=$PWD/datasets/squad
# mkdir -p $SQUAD_DIR
# wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json
# wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json

PWD=$(pwd)
CBT_DIR=$PWD/datasets/cbt
mkdir -p $CBT_DIR
# get CBT data
wget http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz

# unpack all files
tar -zxvf $CBT_DIR/CBTest.tgz -d $CBT_DIR

# move the train/valid/test file of CBT-NE and CBT-CN datasets to CBT_DIR

# Download GloVe
GLOVE_DIR=$PWD/datasets/glove
mkdir -p $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.zip
unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR
