def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    ### create classifier
    nb_classifyer = GaussianNB()

    ### fit the classifier on the training features and labels
    nb_classifyer.fit(features_train, labels_train)
    
    ### return the fit classifier
    return nb_classifyer
    
    ### your code goes here!
