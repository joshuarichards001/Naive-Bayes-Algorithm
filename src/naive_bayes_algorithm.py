import math
import sys
import statistics


# Parses a file and turns it into a list of instances
def file_parser(file_num):
    instances = list()
    with open(sys.argv[file_num]) as f:
        lines = [line.rstrip() for line in f]
    for line in lines:
        instance_value_list = line.split()
        temp = list()
        for value in instance_value_list:
            temp.append(int(value))
        instances.append(temp)
    return instances


# Separate (into a dictionary) the training set given into the different classes that a instance can be
def separate_train_by_class(train_set):
    separated_data = dict()
    for inst in train_set:
        if inst[-1] not in separated_data:  # inst[-1] is the instances class
            separated_data[inst[-1]] = list()
        separated_data[inst[-1]].append(inst)
    return separated_data


# Separate train_set by class then calculates the stats for each row
def info_by_class(train_set):
    separated = separate_train_by_class(train_set)
    col_info = dict()
    for classification, rows in separated.items():
        col_info[classification] = [(statistics.mean(column), statistics.stdev(column), len(column))
                                    for column in zip(*rows)]  # get rows stats
        del col_info[classification][-1]  # remove class stats
    return col_info


# Calculate the probability of each class for a given instance
def calc_class_probabilities(class_model, inst):
    probabilities = dict()
    for classification, class_info in class_model.items():
        probabilities[classification] = class_model[classification][0][2] / len(training_set)
        for idx in range(len(class_info)):
            # Elegant implementation found
            feature_prob = (1 / (math.sqrt(2 * math.pi) * class_info[idx][1])) * \
                           (math.exp(-((inst[idx] - class_info[idx][0]) ** 2 / (2 * class_info[idx][1] ** 2))))
            probabilities[classification] *= feature_prob
            # print("Instance", i, "Class", classification, "Feature", idx, "Probability", feature_prob)
    return probabilities


# Returns the predicted class and its probability for a given instance
def predict(class_model, inst):
    probabilities = calc_class_probabilities(class_model, inst)
    class_prediction = top_probability = -1
    for classification, probability in probabilities.items():
        print("(" + str(bool(classification)), "Probability:"'{0:f}'.format(probability), end=') ')
        if probability > top_probability:
            class_prediction, top_probability = classification, probability
    return class_prediction, top_probability


# Main code
if __name__ == '__main__':
    training_set = file_parser(-2)  # First file
    test_set = file_parser(-1)  # Second file

    model = info_by_class(training_set)
    for i, instance in enumerate(test_set):
        print("Test Instance", i, instance, end=' - ')
        predicted_class, class_probability = predict(model, instance)
        print("Spam =", bool(predicted_class))
