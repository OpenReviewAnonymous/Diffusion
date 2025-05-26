import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assume S_true.shape = (N, T, 2), for example (100000, 100, 2)
S_true = np.load("two_dimension_garch.npy")
N, T, dim = S_true.shape
print(f"Data shape: {S_true.shape}")

# Split (N, T, 2) into X and Y
# X_flat, Y_flat are (N, T) respectively
X_flat = S_true[:, :, 0]
Y_flat = S_true[:, :, 1]

# Horizontally concatenate X and Y => (N, 2T), first T columns are X (all time steps), last T columns are Y (all time steps)
flatten_data = np.hstack([X_flat, Y_flat])  # shape (N, 2T)

# Compute the 2T × 2T correlation matrix
# rowvar=False means each column is a variable
corr_matrix = np.corrcoef(flatten_data, rowvar=False)

print("Correlation matrix shape:", corr_matrix.shape)
# Print the first 5 rows for a quick look
print(corr_matrix[:5, :5])

# Use seaborn to plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, 
    cmap="RdBu_r",   # Red-blue contrast color, can be changed as preferred
    center=0,        # Set 0 as the color center for correlation coefficients
    square=True
)

# Draw separation lines at position T, dividing the 200×200 into four 100×100 blocks
plt.axvline(x=T, color='k', linestyle='--')  # Vertical separation line
plt.axhline(y=T, color='k', linestyle='--')  # Horizontal separation line

plt.title("200 x 200 Correlation Matrix: [XX, XY; YX, YY]")
plt.tight_layout()
plt.show()
