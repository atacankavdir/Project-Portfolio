import numpy as np, random
def mutation(arr,prob,numswi):
    num = len(arr)
    mutarr = arr
    for i in range(0,num):
        dice = random.uniform(0, 1)
        if dice <= prob:
            if mutarr[i] == 0:
                mutarr[i] = 1
            else:
                mutarr[i] = 0                 
    for i in range(0,numswi):
        dice = random.uniform(0, 1)
        if dice <= prob:
            p1 = random.randint(0,num-1)
            p2 = random.randint(0,num-1)
            mutarr[p1], mutarr[p2] = mutarr[p2], mutarr[p1]   
    
    return np.array(mutarr)


def reproduction(par1,par2,l,u):
    num = len(par1)
    child  = par2[0:l] + par1[l:u+1] + par2[u+1:num]  
#    child2 = par1[0:l] + par2[l:u+1] + par1[u+1:num]
    return np.array(child)
#    return np.array(child), np.array(child2)


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

    return sols[sortindex[0],:], sum(sols[sortindex[0],:]*c)



def optimal_knapsack_solver(A,b,c):
    #Create a list of all possible solutions 
    import itertools
    lst = list(map(list, itertools.product([0, 1], repeat=len(c))))
    # Multiply constraints with all possible solutions
    std = np.dot(A,np.transpose(lst)) 

    #Calculate the differences between calculated solutions and constraints
    diff = np.transpose(std) - np.transpose(b)

    #Calculate and append objective functions value and find optimal solution
    sol_list = []
    for i in range(0,len(diff)):
        if ((diff[i][0] > 0) |  (diff[i][1] > 0)) :
            sol_list.append(0)
        else: 
            sol_list.append(sum(lst[i]*c))

    optimal_solution, optimal_value = lst[sol_list.index(max(sol_list))] , max(sol_list)  
    return optimal_value, optimal_solution