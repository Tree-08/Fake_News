import numpy as np
class DecisionTreeClassifier_my():
    def __init__(self, min_sample_split=2, max_depth=2):
        
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

    def build_tree(self, X, y, curr_depth=0):
        y = np.array(y)
        samples, features = X.shape
        
        if samples == 0:
            # Return leaf node with None or some default label
            return {"leaf": True, "leaf_class": None}
        
        data = np.concatenate((X, y.reshape(-1,1)), axis=1)
        
        if samples >= self.min_sample_split and curr_depth <= self.max_depth:
            best = self.best_split(data, samples, features)
            if not best:  # if best_split returned empty dict (no split found)
                leaf_class = self.leaf_class(y)
                return {"leaf": True, "leaf_class": leaf_class}
            left_tree = self.build_tree(best["data_left"][:,:-1], best["data_left"][:,-1], curr_depth+1)
            right_tree = self.build_tree(best["data_right"][:,:-1], best["data_right"][:,-1], curr_depth+1)
            return {
                "feature_index": best["feature_index"],
                "condition": best["condition"],
                "left_tree": left_tree,
                "right_tree": right_tree,
                "info_gain": best["info_gain"]
            }
        
        leaf_class = self.leaf_class(y)
        return {"leaf": True, "leaf_class": leaf_class}

    
    def leaf_class(self, y):
        labels, counts = np.unique(y, return_counts=True)
        max_count_index = np.argmax(counts)
        return labels[max_count_index]
    
    def best_split(self, data, samples, features):
        best = {}
        max_ig = -np.inf

        for fi in range(features):
            feature_vals = data[:,fi]
            u_vals = np.unique(feature_vals)
            for u in u_vals:
                data_left, data_right = self.split(data, fi, u)
                if len(data_left) > 0 and len(data_right) > 0:
                    y,y_left,y_right = data[:,-1], data_left[:,-1], data_right[:,-1]
                    ig = self.info_gain(y, y_left, y_right)
                    if ig>max_ig:
                        best["feature_index"] = fi
                        best["condition"] = u
                        best["data_left"] = data_left
                        best["data_right"] = data_right
                        best["info_gain"] = ig
                        max_ig = ig
        return best
    
    def split(self, data, fi, u):
        data_left = np.array([row for row in data if row[fi]<=u])
        data_right = np.array([row for row in data if row[fi]>u])
        return data_left, data_right
    
    def info_gain(self, y, y_left, y_right):
        wt_l = len(y_left)/len(y)
        wt_r = len(y_right)/len(y)
        gain = self.entropy(y) - (wt_l*self.entropy(y_left) + wt_r*self.entropy(y_right))
        return gain
    
    def entropy(self, y):
        labels = np.unique(y)
        entropy = 0
        for l in labels:
            p = len(y[y==l])/len(y)
            entropy+=(-p)*np.log2(p)
        return entropy
    
    def print_tree(self, tree=None, indent=""):
        if tree is None:
            tree = self.root  # default to the root of the tree

        # If it's a leaf node
        if tree.get("leaf", False):
            print(indent + f"Predict: {tree['leaf_class']}")
        else:
            # Print the decision rule
            print(indent + f"[X{tree['feature_index']} <= {tree['condition']}]")
            print(indent + "├── True:")
            self.print_tree(tree["left_tree"], indent + "│   ")
            print(indent + "└── False:")
            self.print_tree(tree["right_tree"], indent + "    ")

    def predict(self, X):
        predictions = []
        for row in X:
            node = self.root
            while not node.get("leaf", False):
                if row[node["feature_index"]] <= node["condition"]:
                    node = node["left_tree"]
                else:
                    node = node["right_tree"]
            predictions.append(node["leaf_class"])
        return np.array(predictions)
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)