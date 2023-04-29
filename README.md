2.AIM:
For a given set of training data examples stored in a .CSV file, implement and demonstrate the Candidate-Elimination algorithm to output a description of the set of all hypotheses consistent with the training examples.
ALGORITHM:
Candidate Elimination Algorithm:
1. Initialize both specific and general hypotheses.  
S = < ‘ϕ’, ‘ϕ’, ‘ϕ’, ….., ‘ϕ’ >
G = < ‘?’, ‘?’, ‘?’, ….., ’?’>
Depending on the number of attributes.
2. Take the next example, if the taken example is positive make a specific hypothesis to general.
3. If the taken example is negative make the general hypothesis to a more specific hypothesis.
An example to see how the Candidate Elimination Algorithm works.
Initializing both specific and general hypotheses.
G0 = <<?, ?, ?, ?, ?, ?> , <?, ?, ?, ?, ?, ?> , <?, ?, ?, ?, ?, ?>, 
                 <?, ?, ?, ?, ?, ?> , <?, ?, ?, ?, ?, ?> , <?, ?, ?, ?, ?, ? >>   
S0 = < ϕ, ϕ, ϕ, ϕ, ϕ, ϕ>
When the first training example is supplied (in this case, a positive example), the Candidate Elimination method evaluates the S boundary and determines that it is too specific, failing to cover the positive example.
 G1 = G0 = <<?, ?, ?, ?, ?, ?> , <?, ?, ?, ?, ?, ?> , <?, ?, ?, ?, ?, ?>, 
                 <?, ?, ?, ?, ?, ?> , <?, ?, ?, ?, ?, ?> , <?, ?, ?, ?, ?, ? >>  
S1 = < ‘Sunny’, ‘warm’, ‘normal’, ‘strong’, ‘warm ‘, ‘same’>
When the second (also positive) training example is observed, it has a similar effect of generalizing S to S2, while leaving G intact (i.e., G2 = G1 = G0).
G2 = G0 = <<?, ?, ?, ?, ?, ?> , <?, ?, ?, ?, ?, ?> , <?, ?, ?, ?, ?, ?>, 
                 <?, ?, ?, ?, ?, ?> , <?, ?, ?, ?, ?, ?> , <?, ?, ?, ?, ?, ? >>  
S2 = < ‘Sunny’, ‘warm’, ‘?’, ‘strong’, ‘warm ‘, ‘same’>
Similarly, considering the training instance 3, This negative example demonstrates that the version space’s G border is extremely general;
As a result, the hypothesis in the G border must be specialized until it appropriately categorizes this new negative case.
G3 = <<‘Sunny’, ?, ?, ?, ?, ?>,  <?, ‘warm’, ?, ?, ?, ?>, <?, ?, ?, ?, ?, ?>, <?, ?, ?, ?, ?, ?>, <?, ?, ?, ?, ?, ?>, <?, ?, ?, ?, ?, ‘same’>>
S3 = S2 = < ‘Sunny’, ‘warm’, ‘?’, ‘strong’, ‘warm ‘, ‘same’>
The fourth training example, generalizes the version space’s S boundary. It also results in the removal of one G border member, as this one fails to cover the new positive example.
G4 = <<‘Sunny’, ?, ?, ?, ?, ?>,  <?, ‘warm’, ?, ?, ?, ?>>
S4 = <‘Sunny’, ‘warm’, ?, ‘strong’, ?, ?> 
Finally, the result is produced by synchronizing the G4 and S4 algorithms.

The above diagram depicts the whole version space, including the hypotheses bounded by S4 and G4. The order in which the training examples are given has no impact on the learned version space.
 The final hypothesis is, 
G = <[‘Sunny’, ?, ?, ?, ?, ?>, <?, ‘warm’, ?, ?, ?, ?>>
S = <‘Sunny’, ‘warm’, ?, ‘strong’, ?, ?>
Dataset:

Program:
import numpy as np
import pandas as pd
data = pd.DataFrame(data=pd.read_csv('enjoysport.csv'))
concepts = np.array(data.iloc[:,0:-1])
print(concepts)
target = np.array(data.iloc[:,-1])
print(target)
def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("initialization of specific_h and general_h \n")
    general_h = [["?" for i in range(len(specific_h))] for i in
range(len(specific_h))]
    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x]!= specific_h[x]:
                    specific_h[x] ='?'
                    general_h[x][x] ='?'
            if target[i] == "no":
                for x in range(len(specific_h)):
                    if h[x]!= specific_h[x]:
                        general_h[x][x] = specific_h[x]
                    else:
                        general_h[x][x] = '?'
                        print(" steps of Candidate Elimination Algorithm",i+1)
                        print(specific_h)
                        print(general_h)
    indices = [i for i, val in enumerate(general_h) if val ==
['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
        return specific_h, general_h
s_final, g_final = learn(concepts, target)
print("Final Specific_h:", s_final, sep="\n")
print("Final General_h:", g_final, sep="\n")

6.AIM: Write a program to implement Categorical Encoding, One-hot Encoding
DESCRIPTION:
Categorical Encoding is a process where we transform categorical data into numerical data. 
There are many ways to convert categorical values into numerical values. Each approach has its own trade-offs and impact on the feature set. We have 2 main methods: One-Hot-Encoding and Label-Encoder.
Label Encoding
This approach is very simple and it involves converting each value in a column to a number. Let us focus on one categorical column only.
BRIDGE-TYPE
Arch
Beam
Truss
Cantilever
Tied Arch
Suspension
We choose to encode the text values by putting a running sequence for each text values like below:

Disadvantage: The algorithm might misunderstand that data has some kind of hierarchy/order 0 < 1 < 2 … < 6 and might give 6X more weight to ‘Cable’ in calculation then than ‘Arch’ bridge type.
One-Hot Encoder
Though label encoding is straight but it has the disadvantage that the numeric values can be misinterpreted by algorithms as having some sort of hierarchy/order in them. This ordering issue is addressed in another common alternative approach called ‘One-Hot Encoding’. In this strategy, each category value is converted into a new column and assigned a 1 or 0 (notation for true/false) value to the column. 

Though this approach eliminates the hierarchy/order issues but does have the downside of adding more columns to the data set. It can cause the number of columns to expand greatly if you have many unique values in a category column. 
# import required libraries
import pandas as pd
import numpy as np
# creating initial dataframe
bridge_types = ('Arch','Beam','Truss','Cantilever','Tied Arch','Suspension','Cable')
bridge_df = pd.DataFrame(bridge_types, columns=['Bridge_Types'])
# converting type of columns to 'category'
bridge_df['Bridge_Types'] = bridge_df['Bridge_Types'].astype('category')
# Assigning numerical values and storing in another column
bridge_df['Bridge_Types_Cat'] = bridge_df['Bridge_Types'].cat.codes
bridge_df
---------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(bridge_df[['Bridge_Types_Cat']]).toarray())
# merge with main df bridge_df on key values
bridge_df = bridge_df.join(enc_df)
bridge_df

7.AIM:
 Build an Artificial Neural Network by implementing the Back propagation algorithm and test the same using appropriate data sets.
DESCRIPTION:
Backpropagation neural network is used to improve the accuracy of neural network and make them capable of self-learning. Backpropagation means “backward propagation of errors”. Here error is spread into the reverse direction in order to achieve better performance. 
How BPN works?
BPN learns in an iterative manner. In each iteration, it compares training examples with the actual target label. target label can be a class label or continuous value. The backpropagation algorithm works in the following steps:
Initialize Network: BPN randomly initializes the weights. 
Forward Propagate: After initialization, we will propagate into the forward direction. In this phase, we will compute the output and calculate the error from the target output.
Back Propagate Error: For each observation, weights are modified in order to reduce the error in a technique called the delta rule or gradient descent. It modifies weights in a “backward” direction to all the hidden layers.



import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) # two inputs [sleep,study]
y = np.array(([92], [86], [89]), dtype=float) # one output [Expected % in Exams]
X = X/np.amax(X,axis=0) # maximum of X array longitudinally
y = y/100

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
epoch=5000 	#Setting training iterations
lr=0.1 		#Setting learning rate
inputlayer_neurons = 2 		#number of features in data set
hiddenlayer_neurons = 3 	#number of hidden layers neurons
output_neurons = 1 		#number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons)) #weight of the link from input node to hidden node
bh=np.random.uniform(size=(1,hiddenlayer_neurons)) # bias of the link from input node to hidden node
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons)) #weight of the link from hidden node to output node
bout=np.random.uniform(size=(1,output_neurons)) #bias of the link from hidden node to output node


#draws a random range of numbers uniformly of dim x*y
for i in range(epoch):

#Forward Propogation
    hinp1=np.dot(X,wh)
    hinp=hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1=np.dot(hlayer_act,wout)
    outinp= outinp1+ bout
    output = sigmoid(outinp)

#Backpropagation
    EO = y-output
    outgrad = derivatives_sigmoid(output)
    d_output = EO* outgrad
    EH = d_output.dot(wout.T)

#how much hidden layer weights contributed to error
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad

# dotproduct of nextlayererror and currentlayerop
wout += hlayer_act.T.dot(d_output) *lr
     wh += X.T.dot(d_hiddenlayer) *lr

print("Input: \n" + str(X)) 
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)

10.AIM:
Assuming a set of documents that need to be classified, use the naïve Bayesian Classifier model to perform this task. Built-in Java classes/API can be used to write the program. Calculate the accuracy, precision, and recall for your data set.
DESCRIPTION:
Naive Bayes algorithms for learning and classifying text 
Bayes’ Theorem is stated as:

Where,
P(h|D) is the probability of hypothesis h given the data D. This is called the posterior probability.
P(D|h) is the probability of data d given that the hypothesis h was true.
P(h) is the probability of hypothesis h being true. This is called the prior probability of h. P(D) is the probability of the data. This is called the prior probability of D
After calculating the posterior probability for a number of different hypotheses h, and is interested in finding the most probable hypothesis h ∈ H given the observed data D. Any such maximally probable hypothesis is called a maximum a posteriori (MAP) hypothesis.
Bayes theorem to calculate the posterior probability of each candidate hypothesis is hMAP is a MAP hypothesis provided.
Bayes theorem to calculate the posterior probability of each candidate hypothesis is hMAP is a MAP hypothesis provided.

(Ignoring P(D) since it is a constant)
CLASSIFY_NAIVE_BAYES_TEXT (Doc)
Return the estimated target value for the document Doc. ai denotes the word found in the ith position within Doc.
positions ← all word positions in Doc that contain tokens found in Vocabulary
Return VNB, where
                               
                                            
import pandas as pd
msg=pd.read_csv(r"C:\\Users\\priyanka\\Desktop\\naivetext.csv",names=['message','label'])
print('The dimensions of the dataset',msg.shape)
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
y=msg.labelnum
print(X)
print(y)
#splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y)
print ('\n The total number of Training Data :',ytrain.shape)
print ('\n The total number of Test Data :',ytest.shape)
#output of count vectoriser is a sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)
print('\n The words or Tokens in the text documents \n')
print(count_vect.get_feature_names())
df=pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names())
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm,ytrain)
predicted = clf.predict(xtest_dtm)
#printing accuracy, Confusion matrix, Precision and Recall
from sklearn import metrics
print("\n Accuracy of the classifer is",metrics.accuracy_score(ytest,predicted))
print('\n Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('\n The value of Precision' ,
metrics.precision_score(ytest,predicted))
print('\n The value of Recall' ,
metrics.recall_score(ytest,predicted))


AIM:(14)Write a program to Implement Support Vector MachineS.
---------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
# linear data
X = np.array([1, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1])
y = np.array([2, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 3, 8.8, 7.5])
# show unclassified data
plt.scatter(X, y)
plt.show()
# shaping data for training the model
training_X = np.vstack((X, y)).T
training_y = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
# define the model
clf = svm.SVC(kernel='linear', C=1.0)
# train the model
clf.fit(training_X, training_y)
# get the weight values for the linear equation from the trained SVM model
w = clf.coef_[0]
# get the y-offset for the linear equation
a = -w[0] / w[1]
# make the x-axis space for the data points
XX = np.linspace(0, 13)
# get the y-values to plot the decision boundary
yy = a * XX - clf.intercept_[0] / w[1]
# plot the decision boundary
plt.plot(XX, yy, 'k-')
# show the plot visually
plt.scatter(training_X[:, 0], training_X[:, 1], c=training_y)
plt.legend()
plt.show()
