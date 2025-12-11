# Neural-Guided Proof Search

This project combines neural networks with propositional logic.  
A symbolic prover (using Z3 + hand-written inference rules) generates a dataset of shortest proofs for randomly sampled logical states.  
A neural network is then trained to predict the next proof step, effectively learning to imitate and accelerate symbolic reasoning.

---

## Installation

```bash
git clone https://github.com/sferrante/Neuro-Symbolic-Reasoning.git
cd Neuro-Symbolic-Reasoning

pip install numpy
pip install z3-solver
pip install torch --index-url https://download.pytorch.org/whl/cpu
