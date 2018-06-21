from util import entropy, information_gain, partition_classes
import numpy as np 
import ast
import random

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        #self.tree = []
        # Didn't need to initialize
        pass

    def uniq(self,lst):
        last = object()
        for item in lst:
            if item == last:
                continue
            yield item
            last = item

    def sort_and_deduplicate(self,l):
        return list(self.uniq(sorted(l, reverse=True)))


    def get_split_tree(self,X,y):

        if (len(y) == 0):
            return np.array([[-1, -1, np.NaN, np.NaN]])

        elif (len(y) <= 1):
            num_zeros = 0
            num_ones = 0
            output = 0
            for i in range(0,len(y)):
                if (y[i] == 0):
                    num_zeros += 1
                else:
                    num_ones += 1

            if (num_zeros > num_ones):
                output = 0
            else:
                output = 1

            return np.array([[-1, output, np.NaN, np.NaN]])

        elif (len(self.sort_and_deduplicate(y)) == 1):
            return np.array([[-1, np.max(y), np.NaN, np.NaN]])

        else:
            max_ig = 0.00

            feature_split_val = ""
            feature = -2

            for i in range(0, len(X[0])):

                if isinstance(X[0][i], int) or isinstance(X[0][i], float) or isinstance(X[0][i], long):
                    total = 0
                    for j in range(0,len(X)):
                        total += X[j][i]
                    split_val = total / len(X)
                    split_attribute = i
                    x_left, x_right, y_left, y_right = partition_classes(X, y, split_attribute, split_val)
                    ig = information_gain(y, [y_left, y_right])
                    if (ig < max_ig):
                        max_ig = ig
                        feature = split_attribute
                        feature_split_val = str(split_val)

                else:
                    #ri = random.randint(0,len(X)-1)
                    split_val = X[0][i]
                    split_attribute = i
                    x_left, x_right, y_left, y_right = partition_classes(X, y, split_attribute, split_val)
                    ig = information_gain(y, [y_left, y_right])
                    if (ig < max_ig):
                        max_ig = ig
                        feature = split_attribute
                        feature_split_val = split_val

            x_l, x_r, y_l, y_r = partition_classes(X, y, feature, feature_split_val)

            if len(y_l) == len(y):
                num_zeros = 0
                num_ones = 0
                output = 0
                for i in range(0, len(y_l)):
                    if (y_l[i] == 0):
                        num_zeros += 1
                    else:
                        num_ones += 1

                if (num_zeros > num_ones):
                    output = 0
                else:
                    output = 1
                return np.array([[-1, output, np.NaN, np.NaN]])

            elif len(y_r) == len(y):
                num_zeros = 0
                num_ones = 0
                output = 0
                for i in range(0, len(y_r)):
                    if (y_r[i] == 0):
                        num_zeros += 1
                    else:
                        num_ones += 1

                if (num_zeros > num_ones):
                    output = 0
                else:
                    output = 1
                return np.array([[-1, output, np.NaN, np.NaN]])


            lefttree = self.get_split_tree(x_l,y_l)
            righttree = self.get_split_tree(x_r,y_r)
            root  = np.array([[feature, feature_split_val, 1, lefttree.shape[0] + 1]])
            return np.vstack((root, lefttree, righttree))

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)

        self.tree = self.get_split_tree(X,y)

        return self

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        curr_node = 0;
        root = self.tree[curr_node];
        fac, sv, indexL, indexR = root;

        while (fac != -1):

            if isinstance(record[fac],int):

                if (record[fac] <= int(sv)):
                    curr_node = indexL + curr_node;
                    fac, sv, indexL, indexR = self.tree[curr_node];

                else:
                    curr_node = indexR + curr_node;
                    fac, sv, indexL, indexR = self.tree[curr_node];

            elif isinstance(record[fac],float):
                if (record[fac] <= float(sv)):
                    curr_node = indexL + curr_node;
                    fac, sv, indexL, indexR = self.tree[curr_node];

                else:
                    curr_node = indexR + curr_node;
                    fac, sv, indexL, indexR = self.tree[curr_node];

            elif isinstance(record[fac],long):
                if (record[fac] <= long(sv)):
                    curr_node = indexL + curr_node;
                    fac, sv, indexL, indexR = self.tree[curr_node];

                else:
                    curr_node = indexR + curr_node;
                    fac, sv, indexL, indexR = self.tree[curr_node];

            else:
                if (record[fac] == sv):
                    curr_node = indexL + curr_node;
                    fac, sv, indexL, indexR = self.tree[curr_node];

                else:
                    curr_node = indexR + curr_node;
                    fac, sv, indexL, indexR = self.tree[curr_node];

        y_pred = sv
        
        return y_pred

