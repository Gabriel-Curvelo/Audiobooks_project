#!/usr/bin/env python
# coding: utf-8

# # Audiobooks

# The data is from an audiobook application. Logically, it only refers to the audio versions of the books. Each customer in the database has made a purchase at least once, so it is in the database. I am creating a machine learning algorithm based on our available data that can predict whether a customer will buy again from the audiobook company.
# 
# The main idea is that if a customer is unlikely to return, there is no reason to spend money on advertising for them. If we can focus our efforts only on customers who are likely to convert again, we can make big savings. In addition, the model can identify the most important metrics for a customer to return to. Identifying new customers creates value and growth opportunities.
# 
# The .csv file contains the data. There are several variables: Customer ID, Book duration in mins_avg (average of all purchases), Book duration in minutes_sum (sum of all purchases), Price paid_avg (average of all purchases), Price paid_sum (sum of all purchases), Review (a Boolean variable), Review (out of 10), Total minutes heard, Completion (from 0 to 1), Support requests (number) and Last visit minus purchase date (in days).
# 
# So these are the entries (excluding the customer ID, as it is completely arbitrary. It is more like a name than a number).
# 
# Destinations are a Boolean variable (therefore, 0 or 1). We are taking a 2 year period on our entries and the next 6 months as targets. So, in fact, we are predicting whether: based on the last 2 years of activity and engagement, a customer will convert in the next 6 months. 6 months seems like a reasonable time. If they don't convert after 6 months, chances are they went to a competitor or don't like the audiobook's way of digesting information.
# 
# This is a classification problem with two classes: I will not buy and I will buy, represented by 0s and 1s.

# ### The relevant libraries

# In[1]:


import numpy as np
from sklearn import preprocessing

raw_csv_data = np.loadtxt('Audiobooks_data.csv', delimiter = ',')

unscaled_inputs_all = raw_csv_data[:,1:-1]
targets_all = raw_csv_data[:,-1]
import tensorflow as tf


# ### Balance the dataset

# In[2]:


num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] ==0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)
            
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis = 0)
targets_equal_priors = np.delete (targets_all, indices_to_remove, axis=0)


# ### Standardize the inputs

# In[3]:


scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)


# ### Shuffle the data

# In[4]:


shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]


# ### Split the dataset into train, validation, and test

# In[5]:


samples_count = shuffled_inputs.shape[0]

train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)


# ### Save the three datasets in *.npz

# In[6]:


np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)


# ### Data

# In[7]:


npz = np.load('Audiobooks_data_train.npz')

train_inputs = npz['inputs'].astype(np.float)

train_targets = npz['targets'].astype(np.int)

npz = np.load('Audiobooks_data_validation.npz')

validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

npz = np.load('Audiobooks_data_test.npz')

test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)


# ### Model
# Outline, optimizers, loss, early stopping and training

# In[11]:


input_size = 10
output_size = 2

hidden_layer_size = 50
    
model = tf.keras.Sequential([
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    # the final layer make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])


### Optimizer and the loss function

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

### Training

batch_size = 100

max_epochs = 100

# early stopping mechanism
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

# fit the model
model.fit(train_inputs, # train inputs
          train_targets, # train targets
          batch_size=batch_size, # batch size
          epochs=max_epochs, 
          callbacks=[early_stopping], # early stopping
          validation_data=(validation_inputs, validation_targets), # validation data
          verbose = 2 # making sure we get enough information about the training process
          )  


# ## Test the model
# 
# After training on the training data and validating the validation data, we tested the final predictive power of our model by running it on the test data set that the algorithm has never seen before.

# In[9]:


test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)


# In[10]:


print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))


# Using the initial model and hyperparameters given in this notebook, the final test accuracy should be roughly around 91%.
# 
# Note that each time the code is rerun, we get a different accuracy because each training is different. 

# In[ ]:




