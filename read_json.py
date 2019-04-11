from tqdm import tqdm
import spacy
import ujson as json

def read_from_json(file):
    with open(file, 'rb') as rf:
        return json.load(rf)

def print_example(ex):
    counter = 0
    for ch in ex['doc']:
        counter+=1

    for ch in ex['ques']:
        counter += 1
    for ch in ex['ans']:
        counter += 1
    for ch in ex['cans']:
        counter += 1

    print('counter: {}'.format(counter))

file = './datasets/cbt/cbtest_NE_train_v2000ex_1.json'

data = read_from_json(file)

length = len(data)

print('lenght : {}'.format(length))

ex1 = data
print_example(data)


