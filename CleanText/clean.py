from functions import collect_words
from functions import read_file
from functions import tokenize

collect_words('meta.txt')
print tokenize(read_file('meta.txt'))
