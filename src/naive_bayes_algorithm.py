import sys


# A instance within the data (represents a line in the data) has a class and a set of values
class Instance:
    def __init__(self, classification, values):
        self.classification = classification
        self.values = values


# Parses a file and turns it into a list of instances
def file_parser(file_num):
    instances = list()
    with open(sys.argv[file_num]) as f:
        lines = [line.rstrip() for line in f]
    for line in lines[1:]:
        instance_value_list = line.split()
        temp = Instance(instance_value_list[0], list())
        for value in instance_value_list[1:]:
            temp.values.append(value)
        instances.append(temp)
    return instances


# Main code
if __name__ == '__main__':
    training_set = file_parser(-2)  # First file
    test_set = file_parser(-1)  # Second file