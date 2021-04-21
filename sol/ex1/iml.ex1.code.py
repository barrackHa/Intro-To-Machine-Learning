import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

#11:
plot_3d(x_y_z)

#12:
s = np.array([[0.1,0,0],[0,0.5,0],[0,0,2]])
scaled_cov = np.multiply(np.multiply(s,cov),s)
print('12: scaled cov:')
print(scaled_cov)
scaled_data = s @ x_y_z
plot_3d(scaled_data)

#13:
orthogonal_m = get_orthogonal_matrix(3)
print('13: Random orthogonal matrix:')
print(orthogonal_m)
scaled_by_random_ortho = orthogonal_m @ scaled_data
print('13: Scaled data multiplied by Random orthogonal matrix:')
print(scaled_by_random_ortho)
plot_3d(scaled_by_random_ortho)
scaled_ortho_cov = np.dot(orthogonal_m, scaled_cov, orthogonal_m.T)
print('13: And after orthogonal matrix multiplication the cov is:')
print(scaled_ortho_cov)

#14:
x_y_projection = x_y_z[0:2, :]
plot_2d(x_y_projection)

#15:
conditional_distribution = x_y_z[:, (x_y_z[2] > -0.4) & (x_y_z[2] < 0.1)][:-1]
plot_2d(conditional_distribution)

#16:
import numpy as np
EPSILON = 0.25
NSAMPLES = 100000
NTOSSES = 1000
data = np.random.binomial(1, EPSILON, (NSAMPLES, NTOSSES))
m_array = np.array(range(1, NTOSSES+1))
#16 A
sequence_0_means = np.cumsum(data[0, :])/m_array
sequence_1_means = np.cumsum(data[1, :])/m_array
sequence_2_means = np.cumsum(data[2, :])/m_array
sequence_3_means = np.cumsum(data[3, :])/m_array
sequence_4_means = np.cumsum(data[4, :])/m_array

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(m_array, sequence_0_means, "-b", label='Sequence 1')
ax.plot(m_array, sequence_1_means, "-g", label='Sequence 2')
ax.plot(m_array, sequence_2_means, "-y", label='Sequence 3')
ax.plot(m_array, sequence_3_means, "-r", label='Sequence 4')
ax.plot(m_array, sequence_4_means, "-m", label='Sequence 5')
ax.set_xlabel('Number of m (number of samples)')
ax.set_ylabel('Normalized mean of m first samples')
plt.legend()
plt.title('Samples Mean as function of m (m first samples)')
plt.show()

#16 B+C
def chebyshev_estimation(m, epsln):
    return np.minimum(1/(4*m*(epsln**2)), np.ones(m.shape))  # cutoff at 1

def hoeffding_estimation(m, epsln):
    return np.minimum(2*np.exp(-2*m*(epsln**2)), np.ones(m.shape))  # cutoff at 1

EPSILON = [0.5, 0.25, 0.1, 0.01, 0.001]
means = np.cumsum(data, axis=1)/m_array
p = 0.25
means_dist_from_p = np.abs(means - p)
for i in EPSILON:
    cheby = chebyshev_estimation(m_array, i)
    hoeff = hoeffding_estimation(m_array, i)
    larger_than_epsln = np.sum(means_dist_from_p > i, axis=0)/NSAMPLES
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(m_array, cheby, "-b", label='Chebyshev bound')
    ax.plot(m_array, hoeff, "-g", label='Hoeffding bound')
    ax.plot(m_array, larger_than_epsln, ".y", label='Percentage of sequences with distance over epsilon')
    ax.set_xlabel('Number of m (number of samples)')
    ax.set_ylabel('Probability estimation bound')
    plt.legend()
    plt.title('Probability estimation using chebyshev and hoeffding as function of m\n epsilon='+str(i))
    plt.show()