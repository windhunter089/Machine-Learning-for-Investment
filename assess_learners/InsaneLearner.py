""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
A simple wrapper for Bagging learner

Student Name: Trung Pham (replace with your name)  		  	   		  		 			  		 			     			  	 
GT User ID: tpham328 (replace with your User ID)  		  	   		  		 			  		 			     			  	 
GT ID: 903748900 (replace with your GT ID)  	   		  		 			  		 			     			  	 
"""
  		  	   		  		 			  		 			     			  	 
import numpy as np  		  	   		  		 			  		 			     			  	 
import BagLearner as bg
import LinRegLearner as lrl
  		  	   		  		 			  		 			     			  	 
class InsaneLearner(object):
    """  		  	   		  		 			  		 			     			  	 
    This is an InsaneLearner, contains 20 BagLearner
    Each BagLearner is composed of 20 Linear Regression Learner		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    def __init__(self, verbose=False):  		  	   		  		 			  		 			     			  	 
        self.verbose = verbose
        self.learner = [bg.BagLearner(learner = lrl.LinRegLearner, bags = 20,
                                      kwargs = {}, boost = False) for i in range(20)]
  		  	   		  		 			  		 			     			  	 
    def author(self):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        :return: The GT username of the student  		  	   		  		 			  		 			     			  	 
        :rtype: str  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        return "tpham328"  # replace tb34 with your Georgia Tech username
  		  	   		  		 			  		 			     			  	 
    def add_evidence(self, data_x, data_y):  		  	   		  		 			  		 			     			  	 
        for learner in self.learner:
            learner.add_evidence(data_x, data_y)
  		  	   		  		 			  		 			     			  	 
    def query(self, points):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Estimate a set of test points given the model we built.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		  		 			  		 			     			  	 
        :type points: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        :return: The predicted result of the input data according to the trained model  		  	   		  		 			  		 			     			  	 
        :rtype: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        predict = np.zeros((points.shape[0], len(self.learner)))
        for i, learner in enumerate(self.learner):
            predict[:, i] = learner.query(points)
        return np.mean(predict, axis = 1)

  		  	   		  		 			  		 			     			  	 
if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    print("the secret clue is 'zzyzx'")  		  	   		  		 			  		 			     			  	 
