import numpy as np
np.seterr(all='warn')
import random as rd

def cost(fx,y):
	c=(fx-y)**2
	cost=sum(c)
	cost=cost*0.5
	return cost[0]

def sigmoid(x):
	return 1/(1+np.exp(-1*x))

def sigm_dervtv(x):
	return x*(1-x)

def predict(w1,w2,X1,y1):
	correct_class=4
	z_inp=np.dot(X1[i],np.transpose(w1))
	a_inp=sigmoid(z_inp)
	a_inp=a_inp.reshape(1,2)
	a_inp=np.hstack((w2[:,0].reshape((1,1)),a_inp))
	pred_out_z=np.dot(a_inp,np.transpose(w2))
	pred_out_a=sigmoid(pred_out_z)
	if((pred_out_a>0.5 and y1[i][0]==1) or (pred_out_a<0.5 and y1[i][0]==0)):
		correct_class-=1
	return correct_class


x=np.array([[0,0],[0,1],[1,0],[1,1]])
no_samples=x.shape[0]
no_features=x.shape[1]
y=np.array([[0],[1],[1],[0]])

no_hid_nodes=2

#input layer parameters initialization
total_weight=no_hid_nodes*no_features
w_inp=np.zeros((total_weight,1))
for i in range(0,total_weight):
	w_inp[i][0]=rd.uniform(0,1)

	
w_inp=w_inp.reshape((no_hid_nodes,no_features))

#adding bias to weight vector

bias_inp=np.ones((no_hid_nodes,1))
w_inp=np.hstack((bias_inp,w_inp))


print("bias_shape",bias_inp.shape)

l=[]
for i in range(0,2):
	l.append(rd.uniform(0,1))

#hidden layer parameters initialization
no_out_nodes=1
length=no_hid_nodes*no_out_nodes
w_hid=np.zeros((length,1))
for i in range(0,length):
	w_hid[i][0]=rd.uniform(0,1)

w_hid=w_hid.reshape((no_out_nodes,no_hid_nodes))

bias_hid=np.ones((no_out_nodes,1))

w_hid=np.hstack((bias_hid,w_hid))

X=np.hstack((np.ones((no_samples,1)),x))
no_out=1
learning_rate=0.0001


for iterations  in range(0,10):


	for i in range(0,no_samples):
		z_inp=np.dot(X[i],np.transpose(w_inp))
		a_inp=sigmoid(z_inp)
		a_inp=a_inp.reshape(1,2)
		a_inp=np.hstack((bias_hid,a_inp))
		pred_out_z=np.dot(a_inp,np.transpose(w_hid))
		pred_out_a=sigmoid(pred_out_z)
		error_last=np.array(pred_out_a-y[i][0])
		delta_last=error_last*sigm_dervtv(pred_out_z)
		error_first=np.dot(np.transpose(w_hid[:,1:]),delta_last)
		delta_first=error_first*sigm_dervtv(z_inp.reshape(2,1))
		w_hid[:,1:]=w_hid[:,1:]-learning_rate*np.transpose(np.dot(np.transpose(a_inp)[1:,:],delta_last))
		w_hid[:,0]=w_hid[:,0]-learning_rate*delta_last
		bias_hid=w_hid[:,0].reshape(no_out_nodes,1)
		w_inp[:,1:]=w_inp[:,1:]-learning_rate*delta_first*X[i][1:]
		w_inp[:,0]=w_inp[:,0]-learning_rate*delta_first.reshape((2,))
		bias_inp=w_inp[:,0].reshape((no_hid_nodes,1))


print("Updated input layer weights are ",w_inp)
print("Updated hidden layer weights are ",w_hid)
print("Number of correctly classified samples with the updated weights after training is ",predict(w_inp,w_hid,X,y))


