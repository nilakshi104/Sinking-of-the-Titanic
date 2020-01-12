import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Reading the csv files

train_dataset=pd.read_csv('train.csv')
test_dataset=pd.read_csv('test.csv')


## Splitting the datasets into labels and input features

Y_trainset=train_dataset.iloc[:,1]
X_trainset=train_dataset.drop('Survived',axis=1)
X_testset=test_dataset

## Creating 'title' series

X_trainset_title=[i.split(',')[1].split('.')[0].strip() for i in X_trainset['Name']]
X_trainset['Title']=pd.Series(X_trainset_title)
X_testset_title=[i.split(',')[1].split('.')[0].strip() for i in X_testset['Name']]
X_testset['Title']=pd.Series(X_testset_title)
X_trainset['Title']=X_trainset['Title'].replace(['Don','Lady','Jonkheer','the Countess','Capt','Sir','Mme','Dona',
                                                'Ms','Major','Col','Mlle','Rev','Dr'],'Rare')

X_testset['Title']=X_testset['Title'].replace(['Don','Lady','Jonkheer','the Countess','Capt','Sir','Mme','Dona',
                                                'Ms','Major','Col','Mlle','Rev','Dr'],'Rare')

## Creating 'family' series
def family(x):
    if x<2:
        return 'Single'
    elif x==2:
        return 'Couple'
    elif x<=4:
        return 'Smallfamily'
    else:
        return 'Largefamily'


X_trainset['Family']=X_trainset['SibSp']+X_trainset['Parch']+1
X_trainset['Family']=X_trainset['Family'].apply(family)

X_testset['Family']=X_testset['SibSp']+X_testset['Parch']+1
X_testset['Family']=X_testset['Family'].apply(family)

# Substituting the NaN values of 'Embarked' and 'Age' series by mode and median of the respective series
X_trainset['Embarked'].fillna(X_trainset['Embarked'].mode()[0],inplace=True)
X_trainset['Age'].fillna(X_trainset['Age'].median(), inplace=True)

X_testset['Embarked'].fillna(X_testset['Embarked'].mode()[0],inplace=True)
X_testset['Age'].fillna(X_testset['Age'].median(), inplace=True)

# Dropping the unneccessary columns
X_trainset=X_trainset.drop(['PassengerId','Cabin','Name','SibSp','Parch','Ticket'],axis=1)
X_testset_passengerid=X_testset['PassengerId'].values
X_testset=X_testset.drop(['PassengerId','Cabin','Name','SibSp','Parch','Ticket'],axis=1)

# Creating the dummmy variables of the series having labels

# Pclass 
dummies=pd.get_dummies(X_trainset['Pclass'],prefix='Pclass')
X_trainset=pd.concat([X_trainset,dummies],axis=1)
X_trainset=X_trainset.drop('Pclass',axis=1)
# Embarked 
dummies=pd.get_dummies(X_trainset['Embarked'],prefix='Embarked')
X_trainset=pd.concat([X_trainset,dummies],axis=1)
X_trainset=X_trainset.drop('Embarked',axis=1)
# Title 
dummies=pd.get_dummies(X_trainset['Title'],prefix='Title')
X_trainset=pd.concat([X_trainset,dummies],axis=1)
X_trainset=X_trainset.drop('Title',axis=1)
# Sex
dummies=pd.get_dummies(X_trainset['Sex'],prefix='Sex')
X_trainset=pd.concat([X_trainset,dummies],axis=1)
X_trainset=X_trainset.drop('Sex',axis=1)
# Family
dummies=pd.get_dummies(X_trainset['Family'],prefix='Family')
X_trainset=pd.concat([X_trainset,dummies],axis=1)
X_trainset=X_trainset.drop('Family',axis=1)

# Doing the same on testset
# Pclass 
dummies=pd.get_dummies(X_testset['Pclass'],prefix='Pclass')
X_testset=pd.concat([X_testset,dummies],axis=1)
X_testset=X_testset.drop('Pclass',axis=1)
# Embarked 
dummies=pd.get_dummies(X_testset['Embarked'],prefix='Embarked')
X_testset=pd.concat([X_testset,dummies],axis=1)
X_testset=X_testset.drop('Embarked',axis=1)
# Title 
dummies=pd.get_dummies(X_testset['Title'],prefix='Title')
X_testset=pd.concat([X_testset,dummies],axis=1)
X_testset=X_testset.drop('Title',axis=1)
# Sex
dummies=pd.get_dummies(X_testset['Sex'],prefix='Sex')
X_testset=pd.concat([X_testset,dummies],axis=1)
X_testset=X_testset.drop('Sex',axis=1)
# Family
dummies=pd.get_dummies(X_testset['Family'],prefix='Family')
X_testset=pd.concat([X_testset,dummies],axis=1)
X_testset=X_testset.drop('Family',axis=1)

# Converting the Dataframes to numpy arrays
X_train=X_trainset.values

X_test=X_testset.values
X_test=X_test.T

Y_train=Y_trainset.values
Y_train=Y_train.reshape(-1,1)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def initialise_parameters(dim):
    w=np.zeros((dim,1))
    b=0
    return w,b


def forward_propagation(w,b,X,Y):
    m=X.shape[1]
    A=sigmoid(np.dot(w.T,X)+b)
    cost=-1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    cost=np.squeeze(cost)
    return A, cost


def predict(w,b,X):
    m=X.shape[1]
    Y_prediction=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)
    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0,i]<=0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
    return Y_prediction


def backward_propagation(A,X,Y):
    m=X.shape[1]
    dw=1/m*np.dot(X,(A-Y).T)
    db=1/m*np.sum(A-Y)
    return dw,db


def model(X_train,Y_train,X_test,num_iterations,learning_rate):
    w,b=initialise_parameters(X_train.shape[0])
    costs=[]
    acc=[]
    for i in range(num_iterations):
        A,cost=forward_propagation(w,b,X_train,Y_train)
        dw,db=backward_propagation(A,X_train,Y_train)
        w=w-learning_rate*dw
        b=b-learning_rate*db
        Y_prediction_train=predict(w,b,X_train)
        accuracy=100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
        if i%100==0:
            acc.append(accuracy)
            costs.append(cost)
            print('After {} iteration cost : {} and accuracy : {}'.format(i,cost,accuracy))
    acc.append(accuracy)
    Y_prediction_test=predict(w,b,X_test)
    print('Final accuracy on training set :',accuracy) 
    return acc,costs,Y_prediction_test

acc,costs,Y_prediction_test=model(X_train.T, Y_train.T,X_test ,num_iterations = 10000, learning_rate = 0.005)

plt.subplot(1,2,1)
plt.plot(costs,'r')
plt.title('Training cost')
plt.xlabel('Number of iterations')
plt.subplot(1,2,2)
plt.plot(acc,'b')
plt.xlabel('Number of iterations')
plt.title('Training accuracy')
plt.show()

#for submission of kaggle
Y_prediction_test=np.squeeze(Y_prediction_test)
Y_prediction_test=Y_prediction_test.astype('int32')
submission=pd.DataFrame({'PassengerId':X_testset_passengerid,'Survived':Y_prediction_test})
filename='Titanic_prediction.csv'
submission.to_csv(filename,index=False)
