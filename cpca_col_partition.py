import numpy as np
import time
from randomx import getur_data 
from randomx import getur_vars
from numpy.linalg import svd
from scipy.linalg import block_diag

X = getur_data()
U, D, Vt = getur_vars()

#Function Definition 
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

#print(X)

chunk_size = int(input("Enter the number of rows you want each chunk to have: "))

number_of_chunks =  X.shape[0]// chunk_size

X_partitions = [X[chunk_size*i: chunk_size*(i + 1), :] for i in range(number_of_chunks)]

if X.shape[0] % chunk_size != 0:
    X_partitions.append(X[chunk_size*number_of_chunks:, :])

#print(X_partitions)

print(f"Total number of chunks: {len(X_partitions)}")

# start = time.time() #Start timing the SVD computation

U_final, D_final, Vt_final = alg1_rowpartition(X_partitions)



#Step 5: Reconstruct X
X_reconstructed = U_final @ np.diag(D_final) @ Vt_final

# Step 6: Compare
err = np.linalg.norm(X - X_reconstructed)
print(f"Frobenius reconstruction error: {err:.3e}")


# end = time.time()  # End timing the SVD computation
# print(f"SVD on all chunks took {end - start:.3f} seconds")




# print(f"Full SVD took {end_full - start_full:.3f} seconds")

# Z_full_recon = U_full @ np.diag(S_full) @ Vt_full

# # # Now compare both versions
# # err_vs_full = np.linalg.norm(Z_full_recon - Z_reconstructed)
# # print(f" Difference between full SVD and partitioned SVD: {err_vs_full:.3e}")

# #Comparing variables
# recon_error1 = np.linalg.norm(U_final - U)
# recon_error2 = np.linalg.norm(S_final - D)
# recon_error3 = np.linalg.norm(Vt_final - Vt)
# print(f"Relative reconstruction error: {recon_error1:.13f}")
# print(f"Relative reconstruction error: {recon_error2:.13f}")
# print(f"Relative reconstruction error: {recon_error3:.13f}")

# # input: col chunks Z1, 2,.....
# # output: U_full, S, Vt_full....


