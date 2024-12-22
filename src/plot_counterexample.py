import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cm as cm

import numpy as np
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt
#import tikzplotlib

#matplotlib.use('pgf')

import os

p = 4
q = p / (p - 1)

def phi(x, y):
    return (1 / p) * (np.power(np.abs(x), p) + np.power(np.abs(y), p))

def conj_phi(x, y):
    return (1 / q) * (np.power(np.abs(x), q) + np.power(np.abs(y), q))

def grad_phi(x, y):
    dx = np.sign(x) * np.power(np.abs(x), p - 1)
    dy = np.sign(y) * np.power(np.abs(y), p - 1)
    return [dx, dy]

def grad_conj_phi(x, y):
    dx = np.sign(x) * np.power(np.abs(x), q - 1)
    dy = np.sign(y) * np.power(np.abs(y), q - 1)
    return [dx, dy]

def hyperplane(x, v, z):
    return z[1] - (v[0] / v[1]) * (x - z[0])



# x0 = [-5, 2]
# v = [0.5, 1]
#
#
# Z = phi(x0[0] - X, x0[1] - Y)




#x = [-6, 0.71]






# x = np.zeros(2)
# x[0] = x0[0] - grad_conj_phi(v[0], v[1])[0]
# x[1] = x0[1] - grad_conj_phi(v[0], v[1])[1]
#
#
#
# normal = grad_phi(x0[0] - x[0], x0[1] - x[1])
# #D = grad_conj_phi(v[0], v[1])
# normal1 = grad_phi(grad_conj_phi(v[0], v[1])[0], grad_conj_phi(v[0], v[1])[1])
#
#
# normal2= (v[0], v[1])
#
# ax.quiver(x[0], x[1], normal[0], normal[1], scale = 20)
# ax.quiver(x[0], x[1], normal1[0], normal1[1], scale = 15)
#
# level = phi(x0[0] - x[0], x0[1] - x[1])
# CS = ax.contour(X, Y, Z, levels=[level])
#
#
# ax.scatter(x[0], x[1])


#ax.clabel(CS, inline=True, fontsize=10)
#ax.set_title('Simplest default with labels')

#normals = (1.3*np.array([3., 0.5]), 1.3*np.array([3., 0.5]), 1.3*np.array([3., 0.5]),
#           1.3*np.array([3., 0.5]),1.3*np.array([3., 0.5]), 1.3*np.array([3., 0.5]), 1.3*np.array([3., 0.5]))

normals = (
    36 * np.array([3., 0.5]), 36 * np.array([3., 0.5]), 36 * np.array([3., 0.5]),
    36 * np.array([3., 0.5]), 36 * np.array([3., 0.5]), 36 * np.array([3., 0.5]),
           36 * np.array([3., 0.5]))


x0 = np.array([-3., 4.])


fig = plt.figure()
ax = fig.gca()
ax.axis('equal')


plt.scatter(-25, 7, c='k', marker='x')
#ax.text(-10+0.2, 0+0.2, '$x^\star$',fontsize=12)

delta = 0.025
#X, Y = np.meshgrid(np.arange(-35.0, 4.0, delta), np.arange(-20.0, 8.0, delta))

X, Y = np.meshgrid(np.arange(-40.0, 10.0, delta), np.arange(-25.0, 15.0, delta))


plt.scatter(x0[0], x0[1], c='k')

x = np.copy(x0)

i = 0
for v in normals:
    x_new = np.zeros(2) #np.copy(x)

    #print(grad_conj_phi(v[0], v[1])[0], grad_conj_phi(v[0], v[1])[1])
    x_new[0] = x[0] - grad_conj_phi(v[0], v[1])[0]
    x_new[1] = x[1] - grad_conj_phi(v[0], v[1])[1]
    plt.scatter(x_new[0], x_new[1], c='k')
    #print(x_new[0], x_new[1])

    if i <= 1 or i >= 5:
        normal = grad_conj_phi(v[0], v[1])
        #plt.quiver(x_new[0], x_new[1], normal[0], normal[1], linewidths=[0.5])
        plt.quiver(x_new[0], x_new[1], v[0], v[1], linewidths=[0.5])

        Z = phi(x[0] - X, x[1] - Y)
        plt.contour(X, Y, Z, levels=[phi(x[0] - x_new[0], x[1] - x_new[1])], colors='k', linestyles='dashed', alpha=0.5)

        xs = np.arange(x_new[0]-2, x_new[0]+2, 0.025)
        ys = hyperplane(xs, v, x_new)
        plt.plot(xs, ys, c='k')

    x[0] = x_new[0]
    x[1] = x_new[1]

    i+=1

#plt.savefig('visualization.pgf')

#label_fontsize = 'scriptsize'
#axis_parameter_set = {
#    'legend style={font=\\%s, legend cell align=left, align=left, draw=white!15!black}' % label_fontsize,
#    'xticklabel style={font=\\%s}' % label_fontsize, 'yticklabel style={font=\\%s}' % label_fontsize}
#tikzplotlib.save(os.path.join('./', 'euclidean_primal_fejer_monotonicity_iterates.tex'),
#                 extra_axis_parameters=axis_parameter_set)
#tikzplotlib.save(os.path.join('./', 'visualization.tex'),
#                 extra_axis_parameters=axis_parameter_set)



plt.show()