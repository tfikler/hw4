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
      
      # Calculate sigmoid function.
      z = np.dot(X, self.theta)
      e_to_power = np.exp(-z)
      
      return 1 / (1 + e_to_power)
    
    def cost_function(self, X, y):
      
      # Calculate cost function.
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
        
        # Initialize theta's with zeros.
        self.theta = np.zeros((X.shape[1]) + 1)
        
        # Apply bais trick for the data.
        new_data = np.c_[np.ones((X.shape[0],1)),X]
        
        
        for i in range(self.n_iter):
          
          #Calculate the new theta's, according to the sigmoid function.
          self.theta = self.theta - self.eta * np.dot(new_data.T,self.sigmoid(new_data) - y)
          self.thetas.append(self.theta)
          
          # Check convergent
          self.Js.append(self.cost_function(new_data,y))
          if i > 0 and (self.Js[-2] - self.Js[-1] < self.eps): # Checking if the loss value is less than self.eps, if true -> break, else continue.
            break 

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        
        # Apply bais trick for the data.
        new_data = np.c_[np.ones((X.shape[0],1)),X]
        
        # Apply sigmoid function on the data
        data_with_sigmoid = self.sigmoid(new_data)
        
        # Iterate over all the data, and predict the class.
        for i in data_with_sigmoid:
          if i > 0.5:
            preds.append(1)
          else:
            preds.append(0)
            
        return np.array(preds)

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
    number_of_folds = folds
    cv_accuracy = None
    accurencies = []
    # set random seed, and shuffle the data
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Create the folds size.
    fold_size = len(X) // number_of_folds

    # Create the actual data folds.
    foldsX = []
    foldsy = []
    start_idx = 0
    for _ in range(number_of_folds):
      folds = X[start_idx : start_idx + fold_size]
      foldsX.append(folds)
      folds = y[start_idx : start_idx + fold_size]
      foldsy.append(folds)
      start_idx += fold_size
    
    # Train and calculate accurancy according to the folds.
    for i in range(number_of_folds):
      indices = [j for j in range(5) if j != i]  # Indices of folds to concatenate
      X_train = np.concatenate([foldsX[j] for j in indices])
      y_train = np.concatenate([foldsy[j] for j in indices])
      X_val = foldsX[i]
      y_val = foldsy[i]
      
      # Fit the data according to the specific fold.
      algo.fit(X_train,y_train)
      
      # Predict according to the specific fold.
      y_pred = algo.predict(X_val)
      
      # Calculate the accuracy according to the specific fold.
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
        

    def init_params(self, data):
        """
        Initialize distribution params
        """
        self.costs = []
        numberOfSamples= data.shape[0]
        
        # Initialize the mus randomly between the maximum and minimum values of the data 
        self.mus = np.zeros(self.k)
        np.random.choice(numberOfSamples, size=self.k)
        max = np.max(data)
        min = np.min(data)
        self.mus = np.random.uniform(min, max, self.k)
        
        # Initialize the weights equaly
        self.weights = np.ones(self.k) / self.k
        
        # Initalize empty responsibilites numpy array with ones.
        self.responsibilities = np.ones((numberOfSamples, self.k)) / self.k
        
        # Initalize random sigmas.
        self.sigmas = np.random.rand(self.k)
        
        
        

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        # Calculate and update the responsibilites numpy array.
        calc = self.weights*norm_pdf(data,self.mus,self.sigmas)
        calc = calc / calc.sum(axis = 1, keepdims=True)
        self.responsibilities = calc
        
        
    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        
        # Update the weights.
        self.weights = np.sum(self.responsibilities, axis=0) / data.shape[0]
        
        # Update the mus.
        self.mus = np.sum(self.responsibilities * data, axis=0) / np.sum(self.responsibilities, axis = 0)
        
        # Update the sigmas.
        self.sigmas = np.sqrt(np.sum(self.responsibilities * ((data - self.mus) ** 2), axis=0) / np.sum(self.responsibilities, axis = 0))
        
    
    def cost_function(self,data):
      
      cost = 0
      
      # Calculate the cost for each sample in the data.
      for i in range(data.shape[0]):
        cost_per_sample = 0
        sample = data[i]
        for j in range(self.k):
          mean = self.mus[j]
          std = self.sigmas[j]
          weight = self.weights[j]
          g_prob = norm_pdf(sample,mean,std)
          w_g_prob = weight * g_prob
          cost_per_sample -= np.log2(w_g_prob)
        cost += np.log2(cost_per_sample)
      
      return cost
    
     
    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        
        # Initialize the params of the data.
        self.init_params(data)
        
        for i in range(self.n_iter):
          
          # Doing expectation step of the EM algorithm.
          self.expectation(data)
          
          # Doing maximization step of the EM algorithm.
          self.maximization(data)
          
          # Check convergent.
          self.costs.append(self.cost_function(data))
          if i > 0 and np.abs((self.costs[-2] - self.costs[-1])) < self.eps: # Checking if the loss value is less than self.eps, if true -> break, else continue.
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
    pdf = 0
    
    for weight, mu, sigma in zip(weights, mus, sigmas):
        pdf += weight * norm_pdf(data, mu, sigma)
    
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
                
        self.classdict = {}
        mus = {}
        weights = {}
        sigmas = {}
        
        # Get prior probabilities
        self.prior = np.unique(y, return_counts=True)[1] / len(y)
        
        # Iterate over the classes
        for j in range(len(self.prior)):
          self.classdict[j] = {}
          
          # Iterate over the features
          for i in range(X.shape[1]):
            self.classdict[j]['prior'] = self.prior[j]
            
            # Fit the params for each feature in each class.
            fit_for_label = EM(self.k)
            fit_for_label.fit(X[y.flatten() == j][:, i].reshape(-1,1))
            
            # Store for each feature in each class the weights, mus, and sigams.
            mus[i] = fit_for_label.mus
            weights[i] = fit_for_label.weights
            sigmas[i] = fit_for_label.sigmas
          
          # Store the params in a dictionary 
          self.classdict[j]['weights'] = weights
          self.classdict[j]['mus'] = mus
          self.classdict[j]['sigmas'] = sigmas
          
          # Empty the dictionaries to start again.
          mus = {}
          weights = {}
          sigmas = {}
          

        
        
    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        like = 1
        predictions = []
        posterior_probs = []
        
        # Iterate over all sample in the data set.
        for sample in X:
          class_posteriors = []
          
          # Iterate over the classes.
          for label in range(len(self.classdict)):
            
            # Iterate over the features.
            for feature in range(X.shape[1]):
              
              # Under the naive bayes assumption, multiply the features likelihoods.
              like = like * (gmm_pdf(sample[feature], self.classdict[label]['weights'][feature], self.classdict[label]['mus'][feature],self.classdict[label]['sigmas'][feature]))
            
            # calculate the posterior.
            posterior = like * self.prior[label]
            class_posteriors.append(posterior)
            like = 1
          
          # Predict the class according to the max posterior. 
          if (class_posteriors[0] > class_posteriors[1]):
            posterior_probs.append(0)
          else:
            posterior_probs.append(1)   
            
        predictions = posterior_probs
        
        return np.array(predictions)
      

def calculate_accuracy(y_test, y_pred):
  
  # Calculate the total number of elements
  total_elements = y_test.shape[0]
  
  # Compute the number of correct predictions.
  num_matching = np.sum(y_test == y_pred)
  
  # Calculate the accuracy.
  accuracy = (num_matching / total_elements)
  
  return accuracy


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
    
    # Logistic Regresstion.
    lor_model = LogisticRegressionGD(eta = best_eta,eps = best_eps)
    lor_model.fit(x_train, y_train)

    # Calculate accuracy for the Logistic Regression.
    lor_train_acc = calculate_accuracy(y_train,lor_model.predict(x_train))
    lor_test_acc = calculate_accuracy(y_test,lor_model.predict(x_test))

    # Naive Bayes with Gaussian Mixture Model.
    gnb = NaiveBayesGaussian(k = k)
    gnb.fit(x_train, y_train)
    
    # Calculate accuracy for the Naive Bayes.
    bayes_train_acc = calculate_accuracy(y_train, gnb.predict(x_train))
    bayes_test_acc = calculate_accuracy(y_test, gnb.predict(x_test))
    
   
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
    
    mean = [[2, 2, 2]]
    cov = [[[0.4, 0, 0], [0, 0.7, 0], [0, 0, 0.8]], [[1.2, 0, 0.9], [0, 0.2, 0], [0.8, 0, 0.6]]]
    NB_data_class0 = np.random.multivariate_normal(mean[0], cov[0], 250)
    dataset_a_labels1 = np.zeros(250)
    NB_data_class1 = np.random.multivariate_normal(mean[0], cov[1], 250)
    dataset_a_labels2 = np.ones(250)

    dataset_a_features = np.concatenate((NB_data_class0, NB_data_class1))
    dataset_a_labels = np.concatenate((dataset_a_labels1, dataset_a_labels2))
    
    
    mean = [[7, 6.4, 0], [4, 8, 2]]
    cov = [[[1, 0.3, 0], [0.5, 1, 0.8], [0.2, 0.764, 1]], [[2.1, 0, 0.4], [0, 0.2, 0.6], [0, 1.6, 0.6]]]
    LoR_data_class0 = np.random.multivariate_normal(mean[0], cov[0], 250)
    dataset_a_labels1 = np.zeros(250)
    LoR_data_class1 = np.random.multivariate_normal(mean[1], cov[1], 250)
    dataset_a_labels2 = np.ones(250)

    dataset_b_features = np.concatenate((LoR_data_class0, LoR_data_class1))
    dataset_b_labels = np.concatenate((dataset_a_labels1, dataset_a_labels2))
    
    
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }