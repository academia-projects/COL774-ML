import numpy as np
import logistic_regression as lr
import sys
import pandas as pd
import csv




class ObliqueDecisionTree:
    def __init__(self, max_depth, min_samples_split=8):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        #self.weight_csv = []
        
        
        
    
    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        impurity = 1.0 - sum((count / len(y)) ** 2 for count in counts)
        return impurity
    
    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        impurity = -sum((count/len(y))*np.log(count/len(y)) for count in counts)
        return impurity

    
    
    def best_split(self, X, y):
        weights = lr.logistic_regression(X, y)
        #print(weights)
        split_decision = X@weights
        sorted_indices = np.argsort(split_decision)
        #sorted_labels = y[sorted_indices]
        sorted_values = split_decision[sorted_indices]
        #print(sorted_values[0:5])
        

        #Determinig best Threshold
        best_threshold = None
        min_gini = float('inf')
        min_entropy = float('inf')
        best_plst = None
        best_prst = None
        original_gini = self._gini(y)
        original_entropy = self._entropy(y)
        #Iterating over sorted values
        for i in range(len(y)-1):
            threshold = (sorted_values[i] + sorted_values[i+1])/2  #midpoint of consecutive values
            potential_left_subtree = split_decision <= threshold
            potential_right_subtree = split_decision > threshold

            if(len(y[potential_left_subtree])==0 or len(y[potential_right_subtree])==0):
                continue
            

            #gini = (self._gini(y[potential_left_subtree])*len(y[potential_left_subtree]) + self._gini(y[potential_right_subtree])*len(y[potential_right_subtree]))/len(y)
            entropy = (self._entropy(y[potential_left_subtree])*len(y[potential_left_subtree]) + self._entropy(y[potential_right_subtree])*len(y[potential_right_subtree]))/len(y)
            gain = original_entropy - entropy
            #print(gini)
            # if gini < min_gini:
            #     #print(f'gini updated at epoch {i}')
            #     min_gini = gini
            #     best_threshold = threshold
            #     best_plst, best_prst = potential_left_subtree, potential_right_subtree

            if(entropy < min_entropy and gain>0):
                min_entropy = entropy
                best_threshold = threshold
                best_plst, best_prst = potential_left_subtree, potential_right_subtree
            

            
        if best_threshold is None:
            return None
            
        return weights, best_threshold, best_plst, best_prst
        
    
    def tree_build(self, X, y, node_id, depth):
        
        
        if len(np.unique(y)) == 1:
            return y[0]
        
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return np.bincount(y).argmax()
        
        result = self.best_split(X, y)
        if result is None:
            return np.bincount(y).argmax()
        
        weights, threshold, left_idx, right_idx = result
        majority_class = np.bincount(y).argmax()
        #self.weight_csv.append([node_id, weights[0], weights[1], threshold])
        
        print(f'{node_id}, {weights[0]}, {weights[1]}, {threshold}')
        #print(f'{len(y[left_idx])}, {len(y[right_idx])}')
        left_subtree = self.tree_build(X[left_idx], y[left_idx], 2*node_id, depth+1)
        right_subtree = self.tree_build(X[right_idx], y[right_idx], 2*node_id +1, depth+1)
        

        return {'node_id' : node_id, 'weights' : weights, 'threshold' : threshold, 'left_subtree' : left_subtree, 'right_subtree' : right_subtree, 'majority_class' : majority_class}
    
    def fit(self, X, y):
        self.tree = self.tree_build(X, y, 1, 0)

    def single_predict(self, node, x):

        if not isinstance(node, dict):   #leaf node
            return node 
        
        # if(node['node_id'] == i):
        #     print('node not updated')
        split_decision = x@node['weights']
        if(split_decision <= node['threshold']):
            return self.single_predict(node['left_subtree'], x)
        else:
            return self.single_predict(node['right_subtree'], x)

    def predict(self, X):
        return np.array([self.single_predict(self.tree, x) for x in X])
    
    def preorder(self, node, weight_csv):
        if not isinstance(node, dict):
            return
        else:
            l=[]
            l.append(node['node_id'])
            l.extend(node['weights'].tolist())
            l.append(node['threshold'])
            weight_csv.append(l)
            self.preorder(node['left_subtree'], weight_csv)
            self.preorder(node['right_subtree'], weight_csv)
    
    def postorder_pruning(self, X_val, y_val, node, par, subtree):
        if not isinstance(node, dict):
            return node

        if isinstance(node['left_subtree'], dict):
            node['left_subtree'] = self.postorder_pruning(X_val, y_val, node['left_subtree'], node, 'left_subtree')
        
        if isinstance(node['right_subtree'], dict):
            node['right_subtree'] = self.postorder_pruning(X_val, y_val, node['right_subtree'], node, 'right_subtree')

        if(par == None):
            return node

        pre_prune_acc = np.mean(self.predict(X_val) == y_val)
        #print(f"accuracy before pruning node {node['node_id']} is {pre_prune_acc}")
        original_node = node.copy()
        node = original_node['majority_class']
        par[subtree] = node
        
        post_prune_acc = np.mean(self.predict(X_val) == y_val)
        #print(f"accuracy on pruning node {original_node['node_id']} is {post_prune_acc}")

        if(post_prune_acc < pre_prune_acc):
            return original_node
        
        #print(pruned_accuarcy)
        return node
    

    

mode = sys.argv[1]  #train/test


if(mode == 'train'):
    tree_type = sys.argv[2]   #pruned/unpruned
    train_data = sys.argv[3]
    df = pd.read_csv(train_data)   #header = None for sample dataset
    X = (df.iloc[:, :-1]).to_numpy(dtype= np.float64)  
    y = (df.iloc[:, -1]).to_numpy(dtype=np.int64)
    
    weight_csv = []
    if(tree_type == 'unpruned'):
        max_depth = int(sys.argv[4])
        obt = ObliqueDecisionTree(max_depth)
        obt.fit(X, y)
        obt.preorder(obt.tree, weight_csv)
        print(np.mean(obt.predict(X) == y))
        save_weights = sys.argv[5]
        with open(save_weights, "w", newline="") as file:
            writer = csv.writer(file)
            for row in weight_csv:
                writer.writerow(row)
    
    if(tree_type == 'pruned'):
        val_data = sys.argv[4]
        max_depth = int(sys.argv[5])
        save_weights = sys.argv[6]
        obt = ObliqueDecisionTree(max_depth)
        obt.fit(X, y)
        df_val = pd.read_csv(val_data)  #header = None for sample dataset
        X_val = (df_val.iloc[:, :-1]).to_numpy(dtype=np.float64)  
        y_val = (df_val.iloc[:, -1]).to_numpy(dtype=np.int64)
    
        best_accuracy = 0
        pruned_accuarcy = np.mean(obt.predict(X_val) == y_val)
        
        while(best_accuracy < pruned_accuarcy):
            best_accuracy = pruned_accuarcy
            obt.tree = obt.postorder_pruning(X_val, y_val, obt.tree, None, None)
            pruned_accuarcy = np.mean(obt.predict(X_val) == y_val)
            
        print(pruned_accuarcy)    
        
        obt.preorder(obt.tree, weight_csv)
        with open(save_weights, "w", newline="") as file:
            writer = csv.writer(file)
            for row in weight_csv:
                writer.writerow(row)

if (mode == 'test'):
    train_data = sys.argv[2]
    val_data = sys.argv[3]
    test_data = sys.argv[4]
    max_depth = int(sys.argv[5])
    save_pred = sys.argv[6]
    obt = ObliqueDecisionTree(max_depth)
    df = pd.read_csv(train_data)   #header = None for sample dataset
    X = (df.iloc[:, :-1]).to_numpy(dtype= np.float64)  
    y = (df.iloc[:, -1]).to_numpy(dtype=np.int64)
    df_val = pd.read_csv(val_data)  #header = None for sample dataset
    X_val = (df_val.iloc[:, :-1]).to_numpy(dtype=np.float64)  
    y_val = (df_val.iloc[:, -1]).to_numpy(dtype=np.int64)
    df_test = pd.read_csv(test_data)  #header = None for sample dataset
    X_test = (df_test.iloc[:, :-1]).to_numpy(dtype=np.float64)  
    y_test = (df_test.iloc[:, -1]).to_numpy(dtype=np.int64)

    obt.fit(X, y)
    print(f"train acc before pruning is {np.mean(obt.predict(X) == y)}")
    best_accuracy = 0
    pruned_accuarcy = np.mean(obt.predict(X_val) == y_val)
        
    while(best_accuracy < pruned_accuarcy):
        best_accuracy = pruned_accuarcy
        obt.tree = obt.postorder_pruning(X_val, y_val, obt.tree, None, None)
        #obt.dfs_pruning(X_val, y_val)
        pruned_accuarcy = np.mean(obt.predict(X_val) == y_val)

    predictions = obt.predict(X_test).tolist()

    train_acc = np.mean(obt.predict(X) == y)
    val_acc = np.mean(obt.predict(X_val) == y_val)
    test_acc = np.mean(obt.predict(X_test) == y_test)

    print(f"train acc is {train_acc}")
    print(f"val acc is {val_acc}")
    print(f"test acc is {test_acc}")

    with open(save_pred, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(predictions)