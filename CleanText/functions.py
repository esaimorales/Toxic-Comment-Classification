def collect_words(file_name):
    with open(file_name) as f:
        bag = set()
        for line in f:
            for word in line.split():
                bag.add(word)

    print len(bag) 
