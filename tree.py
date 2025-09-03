import numpy as np
from sklearn.base import BaseEstimator
import math 

def is_leaf(node):
    return not (hasattr(node, 'feature_index') and hasattr(node, 'threshold'))

def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    
    if len(y) == 0:
        return 0.0
    
    p = np.mean(y, axis=0)
    
    p = p[p > 0]
    
    H = -np.sum(p * np.log2(p + EPS))
    
    return H

def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    if len(y) == 0:
        return 0.0
    p = np.mean(y, axis=0)
    H = 1-np.sum(p ** 2)
    return H

def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    if len(y) == 0:
        return 0.0
    H = np.sum((y - np.mean(y))**2) / len(y)
    # YOUR CODE HERE
    
    return H

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    if len(y) == 0:
        return 0.0
    H = np.sum(np.abs(y - np.median(y))) / len(y)
    # YOUR CODE HERE
    
    return H

def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot

def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]

class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None

class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        left_mask = X_subset[:, feature_index] < threshold
        
        X_left = X_subset[left_mask]
        y_left = y_subset[left_mask]
        
        X_right = X_subset[~left_mask]
        y_right = y_subset[~left_mask]
        
        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """
        # YOUR CODE HERE
        left_mask = X_subset[:, feature_index] < threshold
        
        y_left = y_subset[left_mask]
        
        y_right = y_subset[~left_mask]
                
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        best_improvement = -np.inf
        feature_index = None
        threshold = None
        criteria, is_classification = self.all_criterions[self.criterion_name]
        
        for feature_idx in range(X_subset.shape[1]):
            feature_data = X_subset[:, feature_idx]
            unique_values = np.unique(feature_data)
            unique_sorted = np.sort(unique_values)
            for th_index in range(len(unique_sorted)-1):
                th = (unique_sorted[th_index] + unique_sorted[th_index + 1]) / 2
                left_mask = feature_data < th
                right_mask = feature_data >= th

                left = y_subset[left_mask]
                right = y_subset[right_mask]
                if len(left) == 0 or len(right) == 0:
                    continue 
                
                score_left = criteria(left)
                score_right = criteria(right)
                total_objects = len(left) + len(right)
                original_score = criteria(y_subset)
                weighted_score = (len(left)/total_objects * score_left) + (len(right)/total_objects * score_right)

                if is_classification == True: 
                    improvement = original_score - weighted_score
                else:
                    improvement = original_score - weighted_score

                if improvement > best_improvement:
                    best_improvement = improvement
                    feature_index = feature_idx
                    threshold = th
        
        if feature_index is None:
            return None, None

        return feature_index, threshold
    

    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        def build_tree(X_subset, y_subset, current_depth=0):
            n_samples = X_subset.shape[0]
            if (current_depth >= self.max_depth) or (n_samples <= self.min_samples_split):
                criteria, is_classification = self.all_criterions[self.criterion_name]
                if is_classification:
                    if y_subset.shape[1] > 1:  
                        proba = np.mean(y_subset, axis=0)
                        leaf_value = np.argmax(proba)
                    else:  
                        unique, counts = np.unique(y_subset, return_counts=True)
                        proba = counts / n_samples
                        leaf_value = unique[np.argmax(proba)]
                else:
                    leaf_value = np.mean(y_subset)
                    proba = 0
                return Node(feature_index=None, threshold=leaf_value, proba=proba)
            else:
                feature_index, threshold = self.choose_best_split(X_subset, y_subset)
                (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
                
                left_node = build_tree(X_left, y_left, current_depth + 1)
                right_node = build_tree(X_right, y_right, current_depth + 1)
                
                node = Node(feature_index=feature_index, threshold=threshold, proba=0)
                node.left_child = left_node
                node.right_child = right_node

                return node

        return build_tree(X_subset, y_subset, 0)

    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        self.classification = self.all_criterions[self.criterion_name][1]
        if self.classification and self.n_classes is None:
            self.n_classes = len(np.unique(y))
        self.root = self.make_tree(X, y)


    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, 1))
        
        for i in range(n_samples):
            node = self.root
            
            while node.left_child is not None:
                if X[i, node.feature_index] <= node.value:
                    node = node.left_child
                else:
                    node = node.right_child

            predictions[i] = node.value
        
        return predictions

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        n_samples = X.shape[0]
        n_classes = self.n_classes
        
        y_predicted_probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            node = self.root
            while node.left_child is not None:
                if X[i, node.feature_index] <= node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
                    
            #тут честно признаюсь мне дипсик сказал что я уебок и если размерность будет 0 (т.е. одно чиселко), то у меня всё полетит
            if node.proba.ndim == 0:  
                proba_vector = np.zeros(self.n_classes)
                proba_vector[int(node.proba)] = 1.0
                y_predicted_probs[i] = proba_vector
            else:  
                y_predicted_probs[i] = node.proba
        
        return y_predicted_probs