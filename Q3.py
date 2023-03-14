
import matplotlib.pyplot as plt
import numpy as np

# Define the true position of xy
xy_true = (0.25, 0.25)

# Define sensor positions
sensors = [(1/np.sqrt(1), 1/np.sqrt(1)),
           (-1/np.sqrt(1), 1/np.sqrt(1)),
           (-1/np.sqrt(1), -1/np.sqrt(1)),
           (1/np.sqrt(1), -1/np.sqrt(1))]

# Define the standard deviations
sigma_x = 0.25
sigma_y = 0.25
sigma_i = 0.01

# Define a function to calculate the Euclidean distance between two points
def dist_points(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Define a function to calculate the measured value from a sensor
def measure_value(sensor_pos):
    measurement = dist_points(sensor_pos, xy_true) + np.random.normal(scale=sigma_i)
    while measurement <= 0:
        measurement = dist_points(sensor_pos, xy_true) + np.random.normal(scale=sigma_i)
    return measurement


# Calculate the measured values from each sensor
sensor_measurements = {s: measure_value(s) for s in sensors}

# Print the measured values
print("Sensor data is:")
for i, s in enumerate(sensor_measurements):
    print(f"Distance of sensor k{i+1} from true position: {sensor_measurements[s]:.3f}")


# Define the contour levels
contour_level = np.arange(0, 300, 10)

# Create a meshgrid
x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
X, Y = np.meshgrid(x, y)

# Define the objective functions for each number of sensors
def f0(x, y):
    return x**2/sigma_x**2 + y**2/sigma_y**2

def f1(x, y):
    return (sensor_measurements[sensors[0]] - dist_points(sensors[0], (x, y)))**2/sigma_i**2 + f0(x, y)

def f2(x, y):
    return f1(x, y) + (sensor_measurements[sensors[2]] - dist_points(sensors[2], (x, y)))**2/sigma_i**2

def f3(x, y):
    return f2(x, y) + (sensor_measurements[sensors[1]] - dist_points(sensors[1], (x, y)))**2/sigma_i**2

def f4(x, y):
    return f3(x, y) + (sensor_measurements[sensors[3]] - dist_points(sensors[3], (x, y)))**2/sigma_i**2


# Define a plotting function for each number of sensors
def plot_function(f, num_sensors):
    Z = f(X, Y)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.contourf(X, Y, Z, levels=contour_level, cmap='coolwarm')
    ax.plot([xy_true[0]], [xy_true[1]], marker='+', markersize=20, color='r', label='True xy', mew=2)
    for i, s in enumerate(sensors[:num_sensors]):
        ax.plot([s[0]], [s[1]], marker='o', markersize=10, color='b', label=f'Sensor {i+1}')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.legend()
    return fig


# Plot the objective functions for each number of sensors
fig0 = plot_function(f0, 0)
fig1 = plot_function(f1, 1)
fig2 = plot_function(f2, 2)
fig3 = plot_function(f3, 3)
fig4 = plot_function(f4, 4)

# Add titles and labels
fig0.suptitle('MAP objective function contours for K=0', fontsize=20)
fig0.axes[0].set_xlabel('X', fontsize=20)
fig0.axes[0].set_ylabel('Y', fontsize=20)

fig1.suptitle('MAP objective function contours for K=1', fontsize=20)
fig1.axes[0].set_xlabel('X', fontsize=20)
fig1.axes[0].set_ylabel('Y', fontsize=20)

fig2.suptitle('MAP objective function contours for K=2', fontsize=20)
fig2.axes[0].set_xlabel('X', fontsize=20)
fig2.axes[0].set_ylabel('Y', fontsize=20)

fig3.suptitle('MAP objective function contours for K=3', fontsize=20)
fig3.axes[0].set_xlabel('X', fontsize=20)
fig3.axes[0].set_ylabel('Y', fontsize=20)

fig4.suptitle('MAP objective function contours for K=4', fontsize=20)
fig4.axes[0].set_xlabel('X', fontsize=20)
fig4.axes[0].set_ylabel('Y', fontsize=20)

plt.show()
# Reference from Rohin Arora
