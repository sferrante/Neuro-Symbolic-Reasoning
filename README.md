## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

pip install numpy
pip install z3-solver

# CPU PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# (or choose the CUDA version based on your system)
