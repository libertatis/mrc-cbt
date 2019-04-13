# mrc-cbt
A Tensorflow implementation of ASReader:
	"Text Comprehension with the Attention Sum Reader Network" (ACL2016) 
	https://arxiv.org/abs/1603.01547
	
The code adopted from https://github.com/NLPLearn/QANet

tensorflow 1.12
python 3.6
tqdm
nltk

step 0:

	download CBT dataset and pre-trained glove file, and put the to the right dir:
	
	datasets/cbt/
				cbtest_NE_train.txt
				cbtest_NE_valid_2000ex.txt
				cbtest_NE_test_2500ex.txt
				cbtest_CN_train.txt
				cbtest_CN_valid_2000ex.txt
				cbtest_CN_test_2500ex.txt
	datasets/glove/
				glove.840B.300d.txt

Step 1:

	convert the .txt file to json:
	python cbt_text2json.py

Step 2:

	prepare the TRRecord file for train/valid/test
	python config.py --mode prepro
	
Step 3:

	python config.py --mode train
	
Step 4:

	python config.py --mode test

Run tensorboard for visualisation.
	
$ tensorboard --logdir=./ --port=9102
	

	
