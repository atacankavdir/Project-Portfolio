import numpy as np, random
from reproduction import *
from mutation import *
from array import *

def GIPSolver(A,b,c,size,time,ratio,prob,l,u):
   cons, var = A.shape
   sols = np.array([np.zeros(var),np.ones(var)])
   # initial population #######################################################
   for i in range(size):
       sol = np.random.choice([0, 1], size=(var,), p=[1/3, 2/3])
       sols = np.append(sols,[sol],axis=0)
   LHS = np.dot(A,np.transpose(sols))    
   for i in range(size):    
       LHS[:,i] = LHS[:,i] - np.transpose(b) 
   LHS = LHS*100
   LHS[LHS>0] = 0
   # penalizing infeasibility better ##########################################
#   for i in range(size): 
#       for j in range(cons): 
#           if LHS[i,j]<0: 
#               LHS[i,j] = LHS[i,j] - 100
   ############################################################################
   score = np.dot(c,np.transpose(sols)) - LHS.sum(axis=0)
   sortindex = np.argsort(score)
   ############################################################################
   for t in range(time):          
   # Reproduction Phase - Crossover ###########################################
       numitem, rep = 0, 1
       children = np.empty([np.int(size*ratio)+1, var])
       while numitem <= np.int(size*ratio): 
           p1 = random.randint(0,np.int(size*ratio)-1)
           if rep == 1:
               p2 = random.randint(0,np.int(size*ratio)-1)
               child = reproduction(sols[p1,:].tolist(),sols[p2,:].tolist(),l,u)
               rep = 2
           else:
               p3 = random.randint(np.int(size*ratio),size-1)
               child = reproduction(sols[p1,:].tolist(),sols[p3,:].tolist(),l,u)
               rep = 1
           children[numitem,:] = child 
#           np.row_stack((children,child))
           numitem += 1
       
       sols = np.delete(sols,sortindex[np.int(size*0.5+1):np.int(size+2)],0)
       sols = np.row_stack((sols,children))
       # recalculation ########################################################
       LHS = np.dot(A,np.transpose(sols))    
       for i in range(LHS.shape[1]):    
           LHS[:,i] = LHS[:,i] - np.transpose(b) 
       LHS = LHS*100
       LHS[LHS>0] = 0
       score = np.dot(c,np.transpose(sols)) - LHS.sum(axis=0)
       sortindex = np.argsort(score)
    
       # Mutation Phase #######################################################
       nmut = 0
       while nmut < np.int(size*ratio):
           ind = random.randint(1,size-2)
           sols[sortindex[ind],:] = mutation(sols[sortindex[ind],:],prob,np.int(size*ratio))
           nmut += 1           
       # Migration Phase ######################################################
       sols = np.unique(sols, axis=0)
       distinct, var = np.shape(sols)
       if distinct < size+2:
           for i in range(size-distinct+2):
               sol = np.random.choice([0, 1], size=(var,), p=[1/3, 2/3])
               sols = np.append(sols,[sol],axis=0) 
       # recalculation        
       LHS = np.dot(A,np.transpose(sols))    
       for i in range(size+2):    
           LHS[:,i] = LHS[:,i] - np.transpose(b) 
       LHS = LHS*100
       LHS[LHS>0] = 0
       score = np.dot(c,np.transpose(sols)) - LHS.sum(axis=0)
       sortindex = np.argsort(score)

   return sols[sortindex[0],:], score[sortindex[0]]    

c = np.array([1, 2, 3, 4])
A = np.array([[1, 2, 2, 1],[2, 2, 2, 1], [3, 1, 0, 1]])
b = [3, 4, 3] 
#def GIPSolver(A,b,c,size,time,ratio,prob,l,u):
sol, score = GIPSolver(A,b,c,10,100,0.2,0.2,1,2)   
print(sol)
print(score)
#   print(np.shape(A))
#   print(np.shape(samplesol))
#   print(np.dot(A,np.transpose(samplesol)))
   
#   newrow = [1,2,3]
#A = numpy.concatenate((A,newrow))