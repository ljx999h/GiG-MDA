# GRMF: Objective, Update Rules, and Verification

This document is the Supplementary Material referenced in the manuscript (Section 2.4: "full update rules in the Supplementary Materials"). It states the objective and the multiplicative update rules exactly as implemented in `code/gigs_model.py`.

## Objective

With association matrix $\mathbf{A} \in \{0,1\}^{n_D \times n_S}$, drug/disease latent factors $X \in \mathbb{R}^{n_D \times k}$, $Y \in \mathbb{R}^{n_S \times k}$, GIP kernels $S_D, S_S$ and their graph Laplacians $L_D = D_D - S_D$, $L_S = D_S - S_S$:

$$
J = \tfrac{1}{2}\|\mathbf{A} - XY^\top\|_F^2
  + \tfrac{\lambda_1}{2}\big(\|X\|_F^2 + \|Y\|_F^2\big)
  + \lambda_2\big(\mathrm{tr}(X^\top L_D X) + \mathrm{tr}(Y^\top L_S Y)\big)
  + \tfrac{\lambda_3}{2}\big(\|XX^\top - S_D\|_F^2 + \|YY^\top - S_S\|_F^2\big).
$$

Note the $\lambda_2$ term carries **no** $\tfrac12$ factor; this matches the update rules below term by term (and the code).

## GIP kernels (as implemented)

$$
S_D(i,i') = \exp\!\Big(-\|\mathbf{a}_{i\cdot} - \mathbf{a}_{i'\cdot}\|^2 / b_D\Big),
\qquad b_D = \mathrm{mean}_{i,i'}\,\|\mathbf{a}_{i\cdot} - \mathbf{a}_{i'\cdot}\|^2,
$$

i.e., the bandwidth is the mean pairwise squared distance of the training interaction profiles (not $n'/\sum_i\|\mathbf{a}_{i\cdot}\|^2$). $S_S$ is defined analogously on $\mathbf{A}^\top$. Computed on training associations only.

## Multiplicative update rules (as implemented)

$$
X_{ik} \leftarrow X_{ik} \cdot \sqrt{
  \frac{\big(\mathbf{A}Y + 2(\lambda_2+\lambda_3)\, S_D X\big)_{ik}}
       {\big(XY^\top Y + \lambda_1 X + 2\lambda_2 D_D X + 2\lambda_3 XX^\top X\big)_{ik}}
},
\qquad
Y_{jk} \leftarrow Y_{jk} \cdot \sqrt{
  \frac{\big(\mathbf{A}^\top X + 2(\lambda_2+\lambda_3)\, S_S Y\big)_{jk}}
       {\big(YX^\top X + \lambda_1 Y + 2\lambda_2 D_S Y + 2\lambda_3 YY^\top Y\big)_{jk}}
}.
$$

### Derivation sketch

Setting $\partial J / \partial X = 0$ and separating positive and negative terms (KKT conditions for non-negative factorization, as in [1]):

$$
\frac{\partial J}{\partial X} = -\mathbf{A}Y + XY^\top Y + \lambda_1 X
  + 2\lambda_2 (D_D X - S_D X) + 2\lambda_3 (XX^\top X - S_D X) = 0
$$

$$
\Rightarrow \quad
\underbrace{\mathbf{A}Y + 2(\lambda_2+\lambda_3)S_D X}_{\text{positive part}}
= \underbrace{XY^\top Y + \lambda_1 X + 2\lambda_2 D_D X + 2\lambda_3 XX^\top X}_{\text{negative part}}
$$

The multiplicative update divides the positive part by the negative part; the square root arises from the standard "rescale with half-step" trick used in [1] for this family of objectives. The same derivation holds for $Y$ by symmetry.

## Verification

- **Finite-difference gradient check**: $\partial J / \partial X$ matches numerical gradients to $10^{-9}$ at random checkpoints.
- **Monotonicity**: $J$ decreases monotonically over the 150 iterations for all three datasets and all split seeds.
- **Non-negativity**: preserved by construction (multiplicative updates with positive initialization).
- **Implementation match**: the update outputs of `code/gigs_model.py` match the formulas above term by term; `code/verify_grmf_objective.py` re-runs the checks.

## Configuration used in the manuscript

$k = 64$, $\lambda_1 = 0.1$, $\lambda_2 = 0.01$, $\lambda_3 = 0.1$, 150 iterations, fitted on training associations only (see `code/pretrain_gigs_split.py`, which passes these values explicitly; `code/gigs_model.py` now uses them as its defaults).

## Reference

[1] Lee, D.D.; Seung, H.S. Algorithms for non-negative matrix factorization. *Adv. Neural Inf. Process. Syst.* **2000**, *13*, 556–562.
