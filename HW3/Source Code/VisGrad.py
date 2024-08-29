import matplotlib.pyplot as plt


class objective_func:
    def eval(self,x,w):
        return (x[0]*w[0] - 5*x[1]*w[1] - 2)**2
    def gradient(self,x,w):
        grad_w1 = 2*(x[0]*w[0] - 5*x[1]*w[1] - 2)*x[0]
        grad_w2 = 2*(x[0]*w[0] - 5*x[1]*w[1] - 2)*(-5*x[1])
        return grad_w1, grad_w2
    
w = [0,0]
learning_rate = 0.01
x = [1, 1]
w1_values, w2_values, J_values = [], [], []

obj_func = objective_func()

for epoch in range(100):
    J = obj_func.eval(x,w)
    w1_values.append(w[0])
    w2_values.append(w[1])
    J_values.append(J)
    grad_w1, grad_w2 = obj_func.gradient(x,w)
    w[0] -= learning_rate * grad_w1
    w[1] -= learning_rate * grad_w2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(w1_values, w2_values, J_values, label='Gradient Descent')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('J')
ax.set_title('3D line plot of w1 vs w2 vs J')
ax.legend()
plt.show()
