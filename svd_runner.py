import sys
import os

# Add the PyParSVD folder to Python path
pyparsvd_parent = os.path.join(os.path.dirname(__file__), 'PyParSVD')
if pyparsvd_parent not in sys.path:
    sys.path.append(pyparsvd_parent)

from pyparsvd.parsvd_serial import ParSVD_Serial

print("✅ PyParSVD loaded manually from pyparsvd/")

if __name__ == "__main__":
    model = ParSVD_Serial()
    print("✅ Successfully created ParSVD_Serial instance!")
