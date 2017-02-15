#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    import numpy as np
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here
    predictions_ = np.array([i[0] for i in predictions])
    ages_ = np.array([i[0] for i in ages])
    net_worths_ = np.array([i[0] for i in net_worths])

    error = pow(net_worths_ - predictions_,2)

    error_tmp = np.sort(error)
    i = int(0.9*len(error_tmp))
    lim_min = min(error_tmp[i:])


    mask = error < lim_min

    cleaned_predictions = predictions_[mask]
    cleaned_age = ages_[mask]
    cleaned_net_worths = net_worths_[mask]
    cleaned_error = error[mask]

    cleaned_data = zip(cleaned_age,cleaned_net_worths, cleaned_error)

    return cleaned_data
