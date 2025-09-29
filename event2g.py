import numpy as np
import matplotlib.pyplot as plt

#Defining variables
TR = 1
hrf = np.array([0, 0, 1, 5, 8, 9.2, 9, 7, 4, 2, 0, -1, -1, -0.8, -0.7, -0.5, -0.3, -0.1, 0, 0])

event_onsets = [1, 9, 11, 20, 25, 31]
max_time = 50
nScans = int(round(max_time / TR))
nBins = len(hrf)

#Establishing empty matrix G
G = np.zeros((nScans, nBins))

for onset in event_onsets:
    onset_idx = onset - 1  # Convert MATLAB's 1-based to Python's 0-based indexing
    if onset_idx + nBins <= nScans:
        G[onset_idx:onset_idx + nBins, :] += np.eye(nBins)

bold_data = G @ hrf + np.random.randn(nScans)

# plt.figure()
# plt.plot(hrf)
# plt.title('HRF')

# plt.figure()
# plt.imshow(G, aspect = 'auto', cmap = 'viridis')
# plt.colorbar()
# plt.title ('Design Matrix G')

# plt.figure()
# plt.plot(bold_data)
# plt.title('Simulated BOLD Signal')

# plt.show()


# Generating predictor weights
P1 = hrf
P2 = hrf*2 
P3 = hrf/31
P4 = hrf + 3

# Computing score matrices using predictor matrices
U1 = G @ P1
U2 = G @ P2
U3 = G @ P3
U4 = G @ P4

# Stack U vectors
U = np.stack([U1, U2, U3, U4], axis=1)  # shape: (nScans, 4)

# Generating a random diagonal matrix
D = np.diag([4, 3, 2, 1])

# Simulate a random orthonormal V (4 x 1000)
np.random.seed(42)  
Vt = np.random.randn(4, 1000)
Vt, _ = np.linalg.qr(Vt.T)  # QR gives orthonormal columns, then transpose back
Vt = Vt.T  # shape: (4, 1000)

Z = U @ D @ Vt

G = np.array(G)
Z = np.array(Z)

def get_data():
    return G, Z

def get_vars():
    return U, D, Vt

if __name__ == "__main__":
    G, Z = get_data()
    print("G and Z loaded.")
    U, D, Vt = get_vars()
    print("U, D and Vt loaded.")





