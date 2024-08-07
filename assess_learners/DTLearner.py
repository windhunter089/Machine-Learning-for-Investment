""""""
"""  		  	   		  		 			  		 			     			  	 
A simple wrapper for Decision Tree Learner	  
Student Name: Trung Pham (replace with your name)  		  	   		  		 			  		 			     			  	 
GT User ID: tpham328 (replace with your User ID)  		  	   		  		 			  		 			     			  	 
GT ID: 903748900 (replace with your GT ID)  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import numpy as np  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
class DTLearner(object):
    """  		  	   		  		 			  		 			     			  	 
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    def __init__(self, leaf_size = 1, verbose=False):
        """  		  	   		  		 			  		 			     			  	 
        Constructor method  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.empty((10000, 4), dtype = object)
  		  	   		  		 			  		 			     			  	 
    def author(self):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        :return: The GT username of the student  		  	   		  		 			  		 			     			  	 
        :rtype: str  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        return "tpham328"  # replace tb34 with your Georgia Tech username
  		  	   		  		 			  		 			     			  	 
    def add_evidence(self, Xtrain, Ytrain):
        """  		  	   		  		 			  		 			     			  	 
        Add training data to learner 
        Xtrain: numpy array of feature X
        Ytrain: numpy 1D array of target variable 		  	   		  		 			  		 			     			  	  	   		  		 			  		 			     			  	 
        """

        self.tree = self.build_DT(Xtrain, Ytrain)
        if self.verbose == True:
             print(self.tree)
             print("tree shape: ",self.tree.shape)


    def build_DT(self, dataX, dataY):
        #if data points < leaf size, take it as a leaf
        if dataX.shape[0] <= self.leaf_size:
            #print(np.array([node_index,"leaf",np.median(dataY),"NA", "NA"]))
            return np.array(["leaf",np.median(dataY),"NA", "NA"])
        #if all Y are the same, that is a leaf
        if len(np.unique(dataY)) == 1:
            #print(np.array([node_index,"leaf", np.median(dataY), "NA", "NA"]))
            return np.array(["leaf",np.median(dataY),"NA", "NA"])


        #determine best feature i to split on based on highest correlation with Y
        best_i = None
        best_correlation = 0
        for i in range(dataX.shape[1]):
            if len(np.unique(dataX[:, i])) == 1:
                continue
            correlation = np.abs(np.corrcoef(dataX[:, i],dataY)[0, 1])
            if np.isnan(correlation):
                continue
            if correlation > best_correlation:
                best_i = i
                best_correlation = correlation

        #Decide splitval and build left and right tree
        SplitVal = np.median(dataX[:, best_i])
        left_tree_shape = dataX[dataX[:, best_i] <= SplitVal].shape[0]
        right_tree_shape = dataX[dataX[:, best_i] > SplitVal].shape[0]
        if (left_tree_shape == dataX.shape[0]) or (right_tree_shape == dataX.shape[0]):
            return np.array(["leaf",np.median(dataY),"NA", "NA"])

        left_tree = self.build_DT(dataX[dataX[:, best_i] <= SplitVal], dataY[dataX[:, best_i] <= SplitVal])
        right_tree = self.build_DT(dataX[dataX[:, best_i] > SplitVal], dataY[dataX[:, best_i] > SplitVal])

        #get left tree number of rows.
        if left_tree.ndim == 1:
            left_tree_rows = 1
        else:
            left_tree_rows = left_tree.shape[0]
        # print("leftree: ", left_tree_rows)
        root = [best_i, SplitVal, 1, left_tree_rows + 1]
        # print(np.vstack((root, left_tree, right_tree)))
        # print("end of tree")
        return np.vstack((root, left_tree, right_tree))


    def query(self, Xtest):
        """
        Estimate a set of test points given the model we built.

        """
        n, m = Xtest.shape
        Ytest = np.zeros(n)
        for i in range(n):
            x = Xtest[i]
            tree = self.tree.copy()
            while tree.shape[0] > 1 and tree[0,0] != "leaf":
                feature = int(float(tree[0,0]))
                SplitVal = float(tree[0,1])
                if x[feature] <= SplitVal:
                    tree = tree[1:, :]
                    # print("lefttrim",tree)
                else:
                    tree = tree[int(float(tree[0,3])):, :]
                    # print("righttrim",tree)

                Ytest[i] = tree[0,1]
            # print("Ytest",i,"is",Ytest[i])
        return Ytest

if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    print("the secret clue is 'zzyzx'")

