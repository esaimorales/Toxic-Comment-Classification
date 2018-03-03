def parse_file(file_name):
    with open(file_name) as f:
        return [[value for value in line.split(',')] for line in f]

a = parse_file('train.csv')
print len(a)
