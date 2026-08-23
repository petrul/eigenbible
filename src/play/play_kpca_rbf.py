import numpy as np
import matplotlib.pyplot as plt

# 1. Create a 2D grid of points (x, y)
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# 2. Reference point at origin x' = (0, 0)
# Calculate squared Euclidean distance: ||x - x'||^2 = X^2 + Y^2
squared_distance = X**2 + Y**2

# 3. Apply RBF Kernel formula with gamma = 0.5
gamma = 0.5
Z = np.exp(-gamma * squared_distance)

# 4. Plot 3D surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

ax.set_title("RBF (Gaussian) Kernel Surface (Centered at 0,0)")
ax.set_xlabel("Feature X1")
ax.set_ylabel("Feature X2")
ax.set_zlabel("Similarity K(x, x')")
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()