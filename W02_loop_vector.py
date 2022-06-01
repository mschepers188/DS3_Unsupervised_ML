import numpy as np
import time

num_features = 150
num_observaties = 50000

print ("Check the difference between loops and vectorized implementation.")
print ("there are {} features and {} observations.".format(num_features, num_observaties))

#theta = np.array([[0.5, 2, 1.3, 2.1]])
theta = np.random.rand(1,num_features)
theta_nav = theta[0]
#data = np.array([ [1,1,2,1], [4,1,4,2],[3,4,1,2],[4,4,3,2] ])
data = np.random.rand(num_observaties, num_features)

m,n = data.shape
X = data[:, :n-1]
y = data[:, [n-1]]
print("y", y.shape)
X = np.c_[np.ones(m), X]
print("X", X.shape)


print ("Loop implementation...")
start_time = int(round(time.time() * 1000))
J_val1 = 0
for i in range(m):
    p = X[i]
    h = 0
    for j in range(len(theta_nav)):
        h += theta_nav[j]*p[j]    #preditced value
    delta = (h - y[i]) ** 2   #squared difference
    J_val1 += delta           #add to total

J_val_nav = J_val1/ (2 * m)  #divide by 2m
end_time = int(round(time.time() * 1000))
print ("Error: {}".format(J_val_nav))
print ("Execution time {} millis.".format(end_time - start_time))



print ("Vectorized implementation")

start_time = int(round(time.time() * 1000))
predictions = np.dot(X, theta.T)
errors = (predictions - y) ** 2
J_val_vec = np.mean(errors)/2

end_time = int(round(time.time() * 1000))
print ("Error: {}".format(J_val_vec) )
print ("Execution time {} millis.".format(end_time - start_time))
