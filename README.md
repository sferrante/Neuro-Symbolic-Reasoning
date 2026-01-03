# Neural-Guided Proof Search

This project combines neural networks with propositional logic.
A symbolic prover (using Z3 + hand-written inference rules) generates a dataset of shortest proofs for randomly sampled logical states.
A neural network is then trained to predict the next proof step, learning to imitate (and potentially accelerate) symbolic reasoning. 

---

## Contents

- `MakeProofData.ipynb` — Generates proof data by sampling random logical states and running the symbolic prover to produce *shortest* proof traces. 
- `ProofAgent.ipynb` — Trains/evaluates a neural “next-step” policy on the generated proof dataset; can be used to test neural-guided proof search.   
- `Z3_Tools.py` — Z3 helpers + hand-written inference rules / utilities used by the symbolic prover.   
- `models.py` — Neural architectures for next-step prediction (and any related training helpers). 
- `utils.py` — Shared utilities (encoding/decoding, batching, metrics, etc.). 
- `Data/` — Stored dataset artifacts (`.npz`) used by the notebooks.  
  - `Data/proof_data_steps.npz` — saved proof traces / supervision targets (next-step labels), typically grouped by proof length.
  - `Data/depths.npz` — saved metadata such as proof depths / length distribution.

---

## Background

Classical symbolic theorem provers are reliable but can be slow because search branching explodes.
A common hybrid idea is: keep the symbolic verifier (so every step is checkable), but learn a neural policy that proposes the most promising next rule/action.

This repo implements that loop for *propositional logic*:
1) generate shortest proof traces with a symbolic prover (Z3 + custom rules),
2) train a neural network to imitate the next proof step,
3) use the neural net to guide proof search.

---

## Results

The figure below shows **out-of-distribution (OOD) generalization** in proof depth: when models are trained only on shallow proofs (e.g. Depth ≤ 1, blue), accuracy drops sharply as we evaluate on deeper proofs (Depth 3–5), indicating a clear distribution shift. 

But as training data includes deeper proofs (Depth ≤ 2,3,4,5), performance recovers to near-perfect accuracy across depths.  **Deep Sets** appears slightly more robust than an MLP under this depth shift, consistent with the fact that it is permutation-invariant and therefore doesn’t waste capacity learning an arbitrary ordering of the “known” statements in the input.

### Accuracy vs Depth
![Accuracy vs Depth](Plots/Accuracy_vs_Depth.png)



---

## Installation

```bash
git clone https://github.com/sferrante/Neuro-Symbolic-Reasoning.git
cd Neuro-Symbolic-Reasoning

pip install numpy
pip install z3-solver
pip install torch --index-url https://download.pytorch.org/whl/cpu
