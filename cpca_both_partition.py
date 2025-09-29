import numpy as np
import time
from randomx import getur_data 
from randomx import getur_vars
from numpy.linalg import svd
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

X = getur_data()
U, D, Vt = getur_vars()

#Function Definition

#Function 1: Row-Partition Only
def alg1_rowpartition(X_partitions):
    U_tilde_blocks = []
    Y_blocks = []

    for X_chunk in X_partitions:
        U_i, D_i, Vt_i = svd(X_chunk, full_matrices=False)

        DVi = np.diag(D_i) @ Vt_i
        Y_blocks.append(DVi)
        U_tilde_blocks.append(U_i)
    
    #Step 1: Stack all D @ V blocks vertically
    Y = np.vstack(Y_blocks)

    #Step 2: Stack all U blocks diagonally
    U_tilde = block_diag(*U_tilde_blocks)

    #Step 3: Performing SVD on Y matrix
    U_y, D_y, Vt_y = svd(Y, full_matrices=False)

    #Step 4: Merging to get the final outputs
    U_final = U_tilde @ U_y
    D_final = D_y
    Vt_final = Vt_y     

    return U_final, D_final, Vt_final

#Function 2: Both Partition
def alg2_bothpartition(X_blocks):

    Z_blocks = []
    V_blocks = []

    s = len(X_blocks)
    k = len(X_blocks[0])

    for j in range(k):
        X_col_blocks = [X_blocks[i][j] for i in range(s)]

        U_j, D_j, Vt_j = alg1_rowpartition(X_col_blocks)

        Z_blocks.append(U_j @ np.diag(D_j))
        V_blocks.append(Vt_j.T)

    V_tilde = block_diag(*V_blocks) 
    Z = np.hstack(Z_blocks)

    U_z, D_z, Vt_z = svd(Z, full_matrices=False)

    V_z = Vt_z.T
    V_final = V_tilde @ V_z
    D_final = D_z
    U_final = U_z
    

    return U_final, D_final, V_final


row_chunk_size = int(input("Enter the number of rows you want each chunk to have: "))
col_chunk_size = int(input("Enter the number of columns you want each chunk to have: "))

number_of_row_chunks =  X.shape[0]// row_chunk_size
if X.shape[0] % row_chunk_size != 0:
    number_of_row_chunks += 1
number_of_col_chunks =  X.shape[1]// col_chunk_size
if X.shape[1] % col_chunk_size != 0:
    number_of_col_chunks += 1

start = time.time () #Start timing the computation
X_blocks = []

for i in range(number_of_row_chunks):
    row_start = i* row_chunk_size
    row_end = min((i + 1) * row_chunk_size, X.shape[0])

    row_blocks = []

    for j in range(number_of_col_chunks):
        col_start = j * col_chunk_size
        col_end = min((j + 1) * col_chunk_size, X.shape[1])
        
        row_blocks.append(X[row_start:row_end, col_start:col_end])
    X_blocks.append(row_blocks)


U2, S2, V2 = alg2_bothpartition (X_blocks)

# reconstruct (V2 is non-transposed, so use V2.T)
X_approx = U2 @ np.diag(S2) @ V2.T

# Checking if this SVD takes lesser time than the column-partition SVD
end = time.time()
print(f"This SVD took {end - start:.3f} seconds")

# Verifying Accuracy (Method 1: Checking if dimensions are the same)
print("Original shape:", X.shape)
print("Reconstructed shape:", X_approx.shape)

# Verifying Accuracy (Method 2: Checking reconstruction error)
recon_error = np.linalg.norm(X - X_approx)
print(f"Relative reconstruction error: {recon_error:.13f}")

#Verifying Accuracy (Method 3: Checking similarity between U, D and V)
recon_error2 = np.linalg.norm(U - U2)
recon_error3 = np.linalg.norm(D - S2)
recon_error4 = np.linalg.norm(Vt - V2.T)
print(f"Relative reconstruction error: {recon_error2:.13f}")
print(f"Relative reconstruction error: {recon_error3:.13f}")
print(f"Relative reconstruction error: {recon_error4:.13f}")

# input: Zij for all i, j; 
# calls the 1st function alg 
# output: U_z, D_z, Vt_z

#another function
#main function: make partition, call that function to get partitions for input to alg functions 

# make sure D is a vector
D_ref = np.diag(D) if D.ndim == 2 else D
V_ref = Vt.T

# simple: sort by descending singular values
idx_ref = np.argsort(-D_ref)
idx_our = np.argsort(-S2)

U_ref = U[:, idx_ref]
D_ref_sorted = D_ref[idx_ref]
V_ref_sorted = V_ref[:, idx_ref]

U2_sorted = U2[:, idx_our]
S2_sorted = S2[idx_our]
V2_sorted = V2[:, idx_our]

# Quick sign fix so U₂ columns roughly line up with U
for k in range(min(U_ref.shape[1], U2_sorted.shape[1])):
    if np.dot(U_ref[:, k], U2_sorted[:, k]) < 0:
        U2_sorted[:, k] *= -1
        V2_sorted[:, k] *= -1

# ---- scatter plots ----
plt.figure()
plt.scatter(U_ref.flatten(), U2_sorted.flatten(), s=6)
plt.xlabel("U (ref)"); plt.ylabel("U₂ (ours)"); plt.title("U vs U₂")

plt.figure()
plt.scatter(D_ref_sorted, S2_sorted, s=12)
plt.xlabel("D (ref)"); plt.ylabel("S₂ (ours)"); plt.title("Singular values")

plt.figure()
plt.scatter(V_ref_sorted.flatten(), V2_sorted.flatten(), s=6)
plt.xlabel("V (ref)"); plt.ylabel("V₂ (ours)"); plt.title("V vs V₂")

plt.show()







