""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import math  		  	   		  		 			  		 			     			  	 
import sys  		  	   		  		 			  		 			     			  	 
import matplotlib.pyplot as plt
import numpy as np  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bg
import InsaneLearner as ins

def experiment_1(train_x, train_y, test_x, test_y):
    #Experiment 1: DT on different leaf size to evaluate overfitting
    training_rmse = []
    testing_rmse = []
    max_leaf = 50
    step = 1

    #loop through each leaf and build decision tree to compare rmse
    for i in range(1, max_leaf, step):
        # create a learner and train it
        learner = dt.DTLearner(leaf_size = i, verbose=False)  # create learner for DT
        learner.add_evidence(train_x, train_y)

        #in sample result
        pred_y = learner.query(train_x)
        in_rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        # print("In sample results ",i)
        # print(f"RMSE: {in_rmse}")

        #out sample result
        pred_y = learner.query(test_x)
        out_rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        # print("Out of sample results ",i)
        # print(f"RMSE: {out_rmse}")

        #append results for plot
        training_rmse.append(in_rmse)
        testing_rmse.append(out_rmse)

    print(learner.author())
    #Plot experiment 1
    x = range(1, max_leaf, step)
    plt.plot(x, training_rmse, label = "rmse of training data")
    plt.plot(x, testing_rmse, label = "rmse of testing data")
    plt.title("Evaluation of overfitting with respect to leaf size in decision tree model")
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.minorticks_on()
    plt.grid(which = 'both', axis = 'x', linestyle = ':')
    plt.legend()
    plt.savefig("figure1.png")
    plt.show()
    plt.clf()

def experiment_2(train_x, train_y, test_x, test_y):
    # Experiment 2: Bagging with fixed # bags and varying leaf size using RT
    training_rmse = []
    testing_rmse = []
    max_leaf = 50
    step = 5
    bags = 5

    #loop through each leaf and build decision tree to compare rmse
    for i in range(1, max_leaf, step):
        # create a learner and train it
        print("Processing max leaf ", i)
        learner = bg.BagLearner(learner = dt.DTLearner,
                                kwargs = {"leaf_size": i},
                                bags = bags,
                                boost = False,
                                verbose=False
                                )  # create bag learner DT
        learner.add_evidence(train_x, train_y)

        #in sample result
        pred_y = learner.query(train_x)
        in_rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        # print("In sample results ",i)
        # print(f"RMSE: {in_rmse}")

        #out sample result
        pred_y = learner.query(test_x)
        out_rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        # print("Out of sample results ",i)
        # print(f"RMSE: {out_rmse}")

        #append results for plot
        training_rmse.append(in_rmse)
        testing_rmse.append(out_rmse)

    print(learner.author())
    #Plot experiment 1
    x = range(1, max_leaf, step)
    plt.plot(x, training_rmse, label = "rmse of training data")
    plt.plot(x, testing_rmse, label = "rmse of testing data")
    plt.title("RMSE with respect to leaf size in 20 bags DT Learner")
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.minorticks_on()
    plt.grid(which = 'both', axis = 'x', linestyle = ':')
    plt.legend()
    plt.savefig("figure2.png")
    plt.show()
    plt.clf()

if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])  		  	   		  		 			  		 			     			  	 

    #remove first column (time value) and first row (column label)
    data = np.array(
        [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]
    )

    #set seed and shuffle data
    np.random.seed(903748900)
    np.random.shuffle(data)

    # compute how much of the data is training and testing  		  	   		  		 			  		 			     			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		  		 			  		 			     			  	 
    test_rows = data.shape[0] - train_rows  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    # separate out training and testing data  		  	   		  		 			  		 			     			  	 
    train_x = data[:train_rows, 0:-1]  		  	   		  		 			  		 			     			  	 
    train_y = data[:train_rows, -1]  		  	   		  		 			  		 			     			  	 
    test_x = data[train_rows:, 0:-1]  		  	   		  		 			  		 			     			  	 
    test_y = data[train_rows:, -1]  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    print(f"{test_x.shape}")  		  	   		  		 			  		 			     			  	 
    print(f"{test_y.shape}")

    experiment_1(train_x, train_y,test_x, test_y)
    experiment_2(train_x, train_y,test_x, test_y)
  		  	   		  		 			  		 			     			  	 


    # learner = bg.BagLearner(
    #     learner = rt.RTLearner,
    #     kwargs = {"leaf_size":10},
    #     bags = 10,
    #     boost = False,
    #     verbose= True
    #)  # create learner for bag


    # learner = ins.InsaneLearner(verbose = False)
   	#
    # # evaluate in sample
    # pred_y = learner.query(train_x)  # get the predictions
    # rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=train_y)
    # print(f"corr: {c[0,1]}")
  	#
    # # evaluate out of sample
    # pred_y = learner.query(test_x)  # get the predictions
    # rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=test_y)
    # print(f"corr: {c[0,1]}")
