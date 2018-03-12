import nltk

def collect_words(file_name):
    with open(file_name) as f:
        bag = set()
        for line in f:
            for word in line.split():
                bag.add(word)

    print len(bag)

def read_file(file_name):
    with open(file_name, 'r') as f:
        data = f.read().replace('\n', '')
    return data

def tokenize(data):
    tokens = nltk.word_tokenize(data)
    return tokens
    
