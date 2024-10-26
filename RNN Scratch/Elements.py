####### Importing Libraries
import os
import torch
import numpy as np

####### Elements

##### Addition
class Add():

    def forward(self,x1,x2):
        return torch.add(x1,x2)
    
    def backward(self,dz):
        dx1 = dz
        dx2 = dz
        return dx1, dx2

##### Multiplication
class Multiply():

    """
    Multiply Class: 

    x1 - Input (N,T,D1)
    x2 - Weight (D1,D2)
    z = x1*x2, (N,T,D2)
    """

    def forward(self,x1,x2):
        return torch.matmul(x1,x2)

    def backward(self,dz,x1,x2):
        dx1 = torch.matmul(dz,torch.t(x2))
        dx2 = torch.sum(torch.bmm(x1.permute(0,2,1),dz),dim=0,keepdim=False)
        return dx1, dx2
    
##### Sigmoid
class sigmoid():

    def forward(self,x):
        return torch.sigmoid(x)
    
    def backward(self,dz,x):

        """
        dl/dz: dz
        x: Input to the sigmoid()
        """
        return torch.mul(dz,torch.mul(torch.sigmoid(x),torch.ones_like(x)-torch.sigmoid(x))) 

##### tanh
class tanh():

    def forward(self,x):
        return torch.tanh(x)
    
    def backward(self,dz,x):

        """
        dl/dz: dz
        x: Input to the tanh()
        """
        return torch.mul(dz,torch.ones_like(x)-torch.square(torch.tanh(x)))
    
##### Softmax
class softmax():

    def forward(self,x):
        
        """
        x: Input logits -> [N,C], C - # of Classes
        """

        return torch.softmax(x,dim=-1)
    
    def backward(self,dz,x):

        """
        dz: dL/dz -> [N,C], C - # of Classes
        x: Input to softmax [N,C] 
        """
        x = self.forward(x)
        batch_size, C = x.size(0),  x.size(-1)
        device = torch.device("cuda:0")
        dout = torch.zeros(batch_size,C,C).to(device)

        for b in range(batch_size):

            for i in range(C):

                for j in range(C):

                    if(i == j):
                        dout[b,i,j] = x[b,i]*(torch.add(1,-x[b,i]))
                    else:
                        dout[b,i,j] = -(x[b,i]*x[b,j])

        return torch.mul(dz,dout)

##### Testing
#device = torch.device("cuda:0")
#soft_max = softmax()
#add = Add()
#sigmoid = Sigmoid()
#a = torch.randn(128,50).to(device)
#b = torch.randn(128,50,50).to(device)
#print(soft_max.forward(a).shape)
#print(soft_max.backward(b,a).shape)
#c = torch.randn(128,100,128).to(device)
#print(add.forward(a,b).shape)
#da, db = add.backward(c)
#print(da.shape,db.shape)
#mul = Multiply()
#x = torch.randn(128,100,65).to(device)
#w = torch.randn(65,85).to(device)
#z = torch.randn(128,100,85).to(device)
#a = mul.forward(x,w)
#print(a.shape)
#dx, dw = mul.backward(z,x,w)
#print(dx.shape, dw.shape)

