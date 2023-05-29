import numpy as np

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def sigmoid(self, X):
      

      z = np.dot(X, self.theta)
      
      e_to_power = np.exp(-z)
      
      return 1 / (1 + e_to_power)
    
    def cost_function(self, X, y):
      
      cost0 = np.dot(y, np.log(self.sigmoid(X)))
      cost1 = np.dot((1-y), np.log(1-self.sigmoid(X)))
      cost = -((cost1 + cost0))/len(y) 
      return cost
    
    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)
        m = X.shape[0]
        self.theta = np.zeros((X.shape[1]) + 1)
        new_data = np.c_[np.ones((X.shape[0],1)),X]
        for i in range(self.n_iter):
          self.theta = self.theta - self.eta * np.dot(new_data.T,self.sigmoid(new_data) - y)
          self.thetas.append(self.theta)
          self.Js.append(self.cost_function(new_data,y))
          if i > 0 and (self.Js[-2] - self.Js[-1] < self.eps): # Checking if the loss value is less than (1e-8), if true -> break, else continue.
            break 

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        new_data = np.c_[np.ones((X.shape[0],1)),X]
        x = self.sigmoid(new_data)
        for i in x:
          if i > 0.5:
            preds.append(1)
          else:
            preds.append(0)
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None
    accurencies = []
    # set random seed
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    fold_size = len(X) // 5

    # Step 3: Assign samples to each fold
    foldsX = []
    foldsy = []
    start_idx = 0
    for _ in range(5):
      folds = X[start_idx : start_idx + fold_size]
      foldsX.append(folds)
      folds = y[start_idx : start_idx + fold_size]
      foldsy.append(folds)
      start_idx += fold_size
    
    for i in range(5):
      indices = [j for j in range(5) if j != i]  # Indices of folds to concatenate
      X_train = np.concatenate([foldsX[j] for j in indices])
      y_train = np.concatenate([foldsy[j] for j in indices])
      X_val = foldsX[i]
      y_val = foldsy[i]
      algo.fit(X_train,y_train)
      y_pred = algo.predict(X_val)
      accuracy = np.mean(y_pred == y_val)
      accurencies.append(accuracy)
    
    cv_accuracy = np.mean(accurencies)   
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    power_of_e = (((data - mu)**2) / (2*(sigma**2)))
    e_to_the_power = np.power(np.e,-power_of_e)
    sqrt = 1 / np.sqrt(2 * np.pi * (sigma**2))
    p = e_to_the_power * sqrt
    
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None
        

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        NF = data.shape[1]
        NS = data.shape[0]
        self.mus = np.ones(2)
        self.mus[0] = data[0]
        self.mus[1] = data[data.shape[0] - 1]
        self.sigmas = np.ones(2)
        self.weights = np.ones(2)
        self.sigmas[0] = 40.3
        self.sigmas[1] = 20.3
        self.weights[0] = 0.5
        self.weights[1] = 0.5
        self.costs = []
        
        

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        calc = self.weights*norm_pdf(data,self.mus,self.sigmas)
        self.responsibilities = calc / np.sum(calc)
        # Normalize the responsibilities
        self.responsibilities /= np.sum(self.responsibilities, axis=1, keepdims=True)
        

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        self.mus = np.sum(self.responsibilities * data, axis = 0) / (data.shape[0]*self.weights) 
        self.sigmas = np.sum(self.responsibilities * ((data - self.mus) ** 2), axis = 0) / (data.shape[0]*self.weights)
        self.weights = np.sum(self.responsibilities, axis = 0) / (data.shape[0])

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        self.costs = []
        for i in range(self.n_iter):
          print(self.mus)
          self.expectation(data)
          self.maximization(data)
          log_likelihood = np.sum(-np.log(self.weights * norm_pdf(data,self.mus,self.sigmas)))
          self.costs.append(log_likelihood)
          if i > 0 and (np.abs(self.costs[-2] - self.costs[-1]) < self.eps): # Checking if the loss value is less than (1e-8), if true -> break, else continue.
            break

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }