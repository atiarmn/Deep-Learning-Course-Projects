import matplotlib.pyplot as plt
import numpy as np


class objective_func:
    def eval(self,x,w):
        return (1/4)*((x*w)**4) - (4/3)*((x*w)**3) + (3/2)*((x*w)**2)
    def gradient(self,x,w):
        return (((x*w)**3) - (4)*((x*w)**2) + (3)*((x*w)))*x
# Section 2 :     
# obj_func = objective_func()
# J = []
# weights = []
# for w in np.arange(-2,5,0.1):
#     J.append(obj_func.eval(1,w))
#     weights.append(w)

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(weights, J, label='Objective function')
# ax.set_xlabel('w1')
# ax.set_ylabel('J')
# ax.set_title('Objective function : w1 vs. J')
# ax.legend()
# plt.show()


#Section 3 and 4:
#Section 3:
# weights = [-1,0.2,0.9,4]
# learning_rate = 0.1
# x = 1
#Section 4:
# w = 0.2
# x = 1 
# learning_rates = [0.001,0.01,1,5]


# obj_func = objective_func()
# for lr in learning_rates:
#     J_values = []
#     for epoch in range(100):
#         try:
#             J = obj_func.eval(x,w)
#             J_values.append(J)
#             gradw = obj_func.gradient(x,w)
#             w -= lr * gradw
#         except:
#             if(len(J_values) >= 1):
#                 J_values.pop()
#                 epoch -=2
#             break

#     w_final = w  
#     J_final = J_values[-1]  

#     epochs = list(range(epoch+1))
#     plt.plot(epochs, J_values)
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.scatter(epoch, J_final, color='purple')  
#     plt.text(epoch, J_final, f'w={w_final},\n J={J_final}', color='purple') 
#     title = 'Gradient descent - initial learning rate = ' + str(lr)
#     plt.title(title)
#     plt.show()

#section 5:
w = 0.2
x = 1 
learning_rate = 5


obj_func = objective_func()

J_values = []
s = 0
r = 0
p1 = 0.9
p2 = 0.999
delta = 10**(-8)

for epoch in range(100):
    J = obj_func.eval(x,w)
    J_values.append(J)
    gradw = obj_func.gradient(x,w)
    s = p1*s + (1 - p1) * gradw
    r = p2*r + (1 - p2) * (gradw*gradw)
    t = epoch + 1
    adam = (s/(1 - p1**t))/(np.sqrt((r/(1 - p2**t))) + delta)
    w -= learning_rate * adam

w_final = w  
J_final = J_values[-1]  
epochs = list(range(epoch+1))
plt.plot(epochs, J_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.scatter(epoch, J_final, color='purple')  
plt.text(epoch, J_final, f'w={w_final},\n J={J_final}', color='purple') 
title = 'Gradient descent with ADAM'
plt.title(title)
plt.show()