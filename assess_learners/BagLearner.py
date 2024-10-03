""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
A simple wrapper for Bagging learner
	   		  		 			  		 			     			  	 
"""
  		  	   		  		 			  		 			     			  	 
import numpy as np  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
class BagLearner(object):
    """  		  	   		  		 			  		 			     			  	 
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    def __init__(self, learner, kwargs, bags, boost = False, verbose=False):
        """  		  	   		  		 			  		 			     			  	 
        Constructor method  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        self.boost = boost
        self.verbose = verbose
        self.learner = []
        for i in range(bags):
            #self.learner = [learner(**kwargs) for i in range(bags)]
            self.learner.append(learner(**kwargs))

    def author(self):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        :return: The GT username of the student  		  	   		  		 			  		 			     			  	 
        :rtype: str  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        return ""  # replace tb34 username
  		  	   		  		 			  		 			     			  	 
    def add_evidence(self, data_x, data_y):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Add training data to learner  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param data_x: A set of feature values used to train the learner  		  	   		  		 			  		 			     			  	 
        :type data_x: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        :param data_y: The value we are attempting to predict given the X data  		  	   		  		 			  		 			     			  	 
        :type data_y: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        for learner in self.learner:
            bag_indices = np.random.choice(data_x.shape[0], data_x.shape[0], replace = True)
            X = data_x[bag_indices, :]
            Y = data_y[bag_indices]
            learner.add_evidence(X, Y)


  		  	   		  		 			  		 			     			  	 
    def query(self, points):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Estimate a set of test points given the model we built.  		  	   		  		 			  		 			     			  	  		 			  		 			     			  	 
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  		 			  		 			     			  	 
        :type points: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        :return: The predicted result of the input data according to the trained model  		  	   		  		 			  		 			     			  	 
        :rtype: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        """
        predicts = np.array([learner.query(points) for learner in self.learner])
        Y = np.mean(predicts, axis = 0)
        # print(Y)
        # print(Y.shape[0])
        return Y


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")  		  	   		  		 			  		 			     			  	 
