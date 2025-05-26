
import numpy as np
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib import rcParams
# rcParams ['font.family'] = 'SimHei'

def single_dimension_garch(T, N):
    # T = 500

    # N = 100 * 1000
    # N = 100 * 1000

    # 1. Data generation: AR(1)-GARCH(1,1)
    mu = 0
    phi = 0.95
    omega = 0.05
    alpha = 0.05
    beta = 0.90

    S_true = np.zeros((N, T))
    for n in tqdm(range(N)):
        x = np.zeros(T)
        epsilon = np.zeros(T)
        sigma2 = np.zeros(T)
        z = np.random.randn(T)
        sigma2[0] = omega / (1 - alpha - beta)
        epsilon[0] = np.sqrt(sigma2[0]) * z[0]
        x[0] = mu + epsilon[0]
        for t in range(1, T):
            sigma2[t] = omega + alpha * epsilon[t-1]**2 + beta * sigma2[t-1]
            epsilon[t] = np.sqrt(sigma2[t]) * z[t]
            x[t] = mu + phi * x[t-1] + epsilon[t]
        S_true[n, :] = x

    return S_true


def two_dimension_garch(T, N):

    T = 100
    N = 100 * 1000

    # 1. Data generation: two-dimensional AR(1)-GARCH(1,1), X leads Y

    # ---------------------------------- #A1
    # (1) Define model parameters
    #   X_t = mu_x + phi_x * X_{t-1} + e_{x,t}
    #   Y_t = mu_y + phi_y * Y_{t-1} + gamma * X_{t-1} + e_{y,t}
    #   Where the volatility of e_{x,t}, e_{y,t} follows their own GARCH(1,1), and e_{x,t}, e_{y,t} are independent
    mu_x, mu_y = 0.0, 0.0
    phi_x, phi_y = 0.95, 0.90
    gamma = 0.8        # Leading effect coefficient of X on Y

    # GARCH(1,1) parameters (can be different for the two variables, or the same)
    omega_x, omega_y = 0.05, 0.05
    alpha_x, alpha_y = 0.05, 0.05
    beta_x,  beta_y  = 0.90, 0.90
    # ---------------------------------- #A1
    
    # Print all parameters
    print("=== Two-dimensional AR(1)-GARCH(1,1) model parameters ===")
    print(f"Number of time steps T: {T}")
    print(f"Number of samples N: {N}")
    print("\nAR(1) parameters:")
    print(f"mu_x: {mu_x}, mu_y: {mu_y}")
    print(f"phi_x: {phi_x}, phi_y: {phi_y}")
    print(f"gamma (leading effect of X on Y): {gamma}")
    print("\nGARCH(1,1) parameters:")
    print(f"omega_x: {omega_x}, omega_y: {omega_y}")
    print(f"alpha_x: {alpha_x}, alpha_y: {alpha_y}")
    print(f"beta_x: {beta_x}, beta_y: {beta_y}")
    print("=== Two-dimensional AR(1)-GARCH(1,1) model parameters ===")

    # ---------------------------------- #A1
    # (2) Reserve array, S_true has shape (N, T, 2), the 3rd dimension stores (X, Y) respectively
    S_true = np.zeros((N, T, 2))
    # ---------------------------------- #A1

    for n in tqdm(range(N)):

        # ---------------------------------- #A1
        # (3) Initialize X, Y and their garch-related quantities
        X = np.zeros(T)
        Y = np.zeros(T)

        epsilon_x = np.zeros(T)
        epsilon_y = np.zeros(T)

        sigma2_x = np.zeros(T)
        sigma2_y = np.zeros(T)

        # Independent normal noise z_x, z_y
        z_x = np.random.randn(T)
        z_y = np.random.randn(T)

        # GARCH steady-state initial value
        sigma2_x[0] = omega_x / (1.0 - alpha_x - beta_x)
        sigma2_y[0] = omega_y / (1.0 - alpha_y - beta_y)

        # t=0 epsilon and X, Y
        epsilon_x[0] = np.sqrt(sigma2_x[0]) * z_x[0]
        epsilon_y[0] = np.sqrt(sigma2_y[0]) * z_y[0]

        # AR(1) initial
        X[0] = mu_x + epsilon_x[0]
        Y[0] = mu_y + epsilon_y[0]
        # ---------------------------------- #A1

        for t in range(1, T):
            # ---------------------------------- #A1
            # (4) Update GARCH volatility σ^2
            sigma2_x[t] = omega_x + alpha_x * (epsilon_x[t-1]**2) + beta_x * sigma2_x[t-1]
            sigma2_y[t] = omega_y + alpha_y * (epsilon_y[t-1]**2) + beta_y * sigma2_y[t-1]

            # Generate new disturbance term epsilon
            epsilon_x[t] = np.sqrt(sigma2_x[t]) * z_x[t]
            epsilon_y[t] = np.sqrt(sigma2_y[t]) * z_y[t]

            # (5) AR(1) + lead-lag relationship
            #   X_t = mu_x + phi_x * X_{t-1} + epsilon_x[t]
            #   Y_t = mu_y + phi_y * Y_{t-1} + gamma * X_{t-1} + epsilon_y[t]
            X[t] = mu_x + phi_x * X[t-1] + epsilon_x[t]
            Y[t] = mu_y + phi_y * Y[t-1] + gamma * X[t-1] + epsilon_y[t]
            # ---------------------------------- #A1

        # ---------------------------------- #A1
        # (6) Store the simulated (X, Y) into the result array
        S_true[n, :, 0] = X
        S_true[n, :, 1] = Y
        # ---------------------------------- #A1

    # Now S_true.shape = (N, T, 2), representing N samples, each of length T, each time point is (X, Y) two dimensions
    # You can choose to save S_true or do further data preprocessing
    return S_true


def plot_correlation_matrix():
    # Assume S_true.shape = (N, T, 2), e.g. (100000, 100, 2)
    S_true = np.load("two_dimension_garch.npy")
    N, T, dim = S_true.shape
    print(f"Data shape: {S_true.shape}")

    # Split (N, T, 2) into X, Y
    # X_flat, Y_flat are (N, T)
    X_flat = S_true[:, :, 0]
    Y_flat = S_true[:, :, 1]

    # Horizontally concatenate X, Y => (N, 2T), first T columns are X at each time, last T columns are Y at each time
    flatten_data = np.hstack([X_flat, Y_flat])  # shape (N, 2T)

    # Compute 2T × 2T correlation matrix
    # rowvar=False means each column is a variable
    corr_matrix = np.corrcoef(flatten_data, rowvar=False)

    print("Correlation matrix shape:", corr_matrix.shape)
    # Print first 5 rows for a quick look
    print(corr_matrix[:5, :5])

    # Use seaborn to plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        cmap="RdBu_r",   # Red-blue contrast, can be changed as you like
        center=0,        # Set 0 as the color center for correlation coefficients
        square=True
    )

    # Draw dividing lines at T to split 200×200 into four 100×100 blocks
    plt.axvline(x=T, color='k', linestyle='--')  # Vertical dividing line
    plt.axhline(y=T, color='k', linestyle='--')  # Horizontal dividing line

    plt.title("200 x 200 Correlation Matrix: [XX, XY; YX, YY]")
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300)
    plt.show()

# Analyze why the correlation matrix between X and Y looks like a diagonal matrix
def analyze_xy_correlation():
    print("\n=== Analysis of correlation between X and Y ===")
    S_true = np.load("two_dimension_garch.npy")
    N, T, dim = S_true.shape
    
    # Extract X and Y data
    X_flat = S_true[:, :, 0]  # (N, T)
    Y_flat = S_true[:, :, 1]  # (N, T)
    
    # Compute correlation matrix between X and Y
    corr_xy = np.zeros((T, T))
    for i in tqdm(range(T)):
        for j in range(T):
            corr_xy[i, j] = np.corrcoef(X_flat[:, i], Y_flat[:, j])[0, 1]
    
    # Plot correlation matrix between X and Y
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_xy,
        cmap="RdBu_r",
        center=0,
        square=True
    )
    plt.title("Correlation Matrix between X and Y")
    plt.xlabel("Time Steps of Y")
    plt.ylabel("Time Steps of X")
    plt.savefig('xy_correlation.png', dpi=300)
    
    # Analyze correlation on the diagonal
    diag_corrs = np.diag(corr_xy)
    plt.figure(figsize=(10, 6))
    plt.plot(range(T), diag_corrs)
    plt.title("Contemporaneous Correlation between X_t and Y_t")
    plt.xlabel("Time Step t")
    plt.ylabel("Correlation Coefficient")
    plt.grid(True)
    plt.savefig('xy_diagonal_correlation.png', dpi=300)
    
    # Analyze lead correlation of X over Y
    lead_corrs = [np.mean([corr_xy[t-lag, t] for t in range(lag, T)]) for lag in range(1, 11)]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), lead_corrs)
    plt.title("Average Lead Correlation of X over Y")
    plt.xlabel("Lead Steps")
    plt.ylabel("Average Correlation Coefficient")
    plt.grid(True)
    plt.savefig('xy_lead_correlation.png', dpi=300)
    
    print("Reasons why the correlation matrix between X and Y looks like a diagonal matrix:")
    print("1. In the model setup, Y_t depends on X_{t-1}, which creates temporal dependence")
    print("2. Due to the AR(1) process, the correlation between X_t and X_{t+k} weakens as k increases")
    print("3. Similarly, the correlation between Y_t and Y_{t+k} also weakens as k increases")
    print("4. When we look at the correlation between X_t and Y_{t+k}, the strongest relationship appears near k=1")
    print("5. This pattern makes the correlation matrix have higher values near the diagonal, forming a visual effect similar to a diagonal matrix")
    print("=== Analysis complete ===")

    # ---------------------------------- #

    # Check if the correlation matrix is really a diagonal matrix
    print("\nCheck if the correlation matrix is really a symmetric matrix:")
    # Check symmetry between XY and YX
    # Create YX correlation matrix (transpose of XY matrix)
    corr_yx = corr_xy.T
    
    # Compute the difference between XY and YX
    diff_matrix = np.abs(corr_xy - corr_yx)
    max_diff = np.max(diff_matrix)
    mean_diff = np.mean(diff_matrix)
    
    print(f"Maximum difference between XY and YX correlation matrices: {max_diff:.6f}")
    print(f"Mean difference between XY and YX correlation matrices: {mean_diff:.6f}")
    
    # Visualize the difference matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        diff_matrix,
        cmap="viridis",
        square=True
    )
    plt.title("Difference Between XY and YX Correlation Matrices")
    plt.xlabel("Time Steps of Y")
    plt.ylabel("Time Steps of X")
    plt.savefig('xy_yx_difference.png', dpi=300)
    
    # Determine if it is approximately symmetric
    is_symmetric = max_diff < 1e-10
    print(f"Are XY and YX correlation matrices symmetric: {'Yes' if is_symmetric else 'No'}")
    
    
    # ---------------------------------- #
    
    
    # Check if XX and YY correlation matrices are symmetric
    print("\nCheck if XX correlation matrix is symmetric:")
    # Assume S_true.shape = (N, T, 2), e.g. (100000, 100, 2)
    S_true = np.load("two_dimension_garch.npy")
    N, T, dim = S_true.shape
    
    # Split (N, T, 2) into X, Y
    # X_flat, Y_flat are (N, T)
    X_flat = S_true[:, :, 0]
    Y_flat = S_true[:, :, 1]
    
    # Horizontally concatenate X, Y => (N, 2T), first T columns are X at each time, last T columns are Y at each time
    flatten_data = np.hstack([X_flat, Y_flat])  # shape (N, 2T)
    
    # Compute 2T × 2T correlation matrix
    # rowvar=False means each column is a variable
    corr_matrix = np.corrcoef(flatten_data, rowvar=False)
    
    # Extract XX correlation matrix
    corr_xx = corr_matrix[:T, :T]
    
    # Compute the difference between XX and its transpose
    diff_xx = np.abs(corr_xx - corr_xx.T)
    max_diff_xx = np.max(diff_xx)
    mean_diff_xx = np.mean(diff_xx)
    
    print(f"Maximum difference in XX correlation matrix: {max_diff_xx:.6f}")
    print(f"Mean difference in XX correlation matrix: {mean_diff_xx:.6f}")
    
    # Visualize XX difference matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        diff_xx,
        cmap="viridis",
        square=True
    )
    plt.title("Difference Between XX and Its Transpose")
    plt.xlabel("Time Steps of X")
    plt.ylabel("Time Steps of X")
    plt.savefig('xx_symmetry_difference.png', dpi=300)
    
    # Determine if XX is approximately symmetric
    is_xx_symmetric = max_diff_xx < 1e-10
    print(f"Is XX correlation matrix symmetric: {'Yes' if is_xx_symmetric else 'No'}")
    
    # Check if YY correlation matrix is symmetric
    print("\nCheck if YY correlation matrix is symmetric:")
    # Extract YY correlation matrix
    corr_yy = corr_matrix[T:, T:]
    
    # Compute the difference between YY and its transpose
    diff_yy = np.abs(corr_yy - corr_yy.T)
    max_diff_yy = np.max(diff_yy)
    mean_diff_yy = np.mean(diff_yy)
    
    print(f"Maximum difference in YY correlation matrix: {max_diff_yy:.6f}")
    print(f"Mean difference in YY correlation matrix: {mean_diff_yy:.6f}")
    
    # Visualize YY difference matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        diff_yy,
        cmap="viridis",
        square=True
    )
    plt.title("Difference Between YY and Its Transpose")
    plt.xlabel("Time Steps of Y")
    plt.ylabel("Time Steps of Y")
    plt.savefig('yy_symmetry_difference.png', dpi=300)
    
    # Determine if YY is approximately symmetric
    is_yy_symmetric = max_diff_yy < 1e-10
    print(f"Is YY correlation matrix symmetric: {'Yes' if is_yy_symmetric else 'No'}")

def check_symmetric_corr_matrix():
    S_true = np.load("two_dimension_garch.npy")
    N, T, dim = S_true.shape
    
    # Split (N, T, 2) into X, Y
    # X_flat, Y_flat are (N, T)
    X_flat = S_true[:, :, 0]
    Y_flat = S_true[:, :, 1]
    
    # Horizontally concatenate X, Y => (N, 2T), first T columns are X at each time, last T columns are Y at each time
    flatten_data = np.hstack([X_flat, Y_flat])  # shape (N, 2T)
    
    # Compute 2T × 2T correlation matrix
    # rowvar=False means each column is a variable
    corr_matrix = np.corrcoef(flatten_data, rowvar=False)
    # Check if this is symmetric
    corr_matrix_symmetric = corr_matrix == corr_matrix.T
    print(f"corr_matrix_symmetric: {corr_matrix_symmetric}")
    # Check if the correlation matrix is symmetric
    is_symmetric = np.allclose(corr_matrix, corr_matrix.T, atol=1e-10)
    print(f"Is the correlation matrix symmetric: {'Yes' if is_symmetric else 'No'}")
    
    # Compute the maximum and mean difference
    diff = np.abs(corr_matrix - corr_matrix.T)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Maximum difference in correlation matrix: {max_diff:.6e}")
    print(f"Mean difference in correlation matrix: {mean_diff:.6e}")
   

TIME_LENGTH = 100
# Generate single-dimensional GARCH data (T=500)
print("Generating single-dimensional GARCH data (T=500, N=100000)...")
data_500 = single_dimension_garch(T=TIME_LENGTH, N=10 * 1000)
# data_500 = single_dimension_garch(T=TIME_LENGTH, N=100 * 1000)
# Save the generated data
print(f"data_{TIME_LENGTH}.shape", data_500.shape)
file_path_500 = f"ar1_garch1_1_data_{TIME_LENGTH}.npy"
np.save(file_path_500, data_500)
print(f"Generated and saved data with T={TIME_LENGTH} to {file_path_500}")
