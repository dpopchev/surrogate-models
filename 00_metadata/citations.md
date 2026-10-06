# Citations of the revised paper structure, mapped to refs.bib (W-059)

The revised structure cites by number ([1] to [53]); refs.bib carries no numbers, and the
numbers are not one paper each ([36] stands for both the Huber loss and AdamW with cosine
annealing, [35] for both GELU and residual networks). So the mapping is per citation site, by
what the site claims. The paper cites keys, never numbers.

Status: **matched** -- the key's title states the claim; **candidate** -- plausible, the
developer confirms; **missing** -- no refs.bib entry fits, the column says what is needed.
This file is not compiled (the paper build copies only .tex and .bib).

| section | claim | outline numbers | refs.bib keys | status |
|---|---|---|---|---|
| Abstract | Bayesian inference needs 1e5 to 1e7 solver evaluations | [3, 4] | Reed2024 | candidate |
| Abstract | vacuum curvature-induced scalarized black holes | [1] | Doneva2017_BH | matched |
| Abstract | matter-coupled scalarized neutron stars | [5] | Doneva2017_NS | matched |
| 1 | probing strong-field GR with compact objects in sTGB gravity | [6, 7] | -- | missing: reviews of strong-field tests / sTGB compact objects |
| 1 | spontaneous scalarization, R_GB as an effective mass | [1, 5] | Doneva2017_BH, Doneva2017_NS | matched |
| 1 | historical spontaneous scalarization in scalar-tensor theory | -- | DamourEspositoFarese1993 | candidate |
| 1 | high-fidelity ODE shooting takes seconds to minutes per evaluation | [8, 9] | -- | missing: the solver papers behind the datasets, or a cost statement |
| 1 | Bayesian estimation impractically slow with the solver | [4, 10] | Reed2024 | candidate |
| 1 | prior neural surrogates for neutron stars | -- | Liodis2024, Kaushikk2026, Liu2026 | candidate |
| 2.1 | ESTGB theories: scalar coupled to R_GB | [11, 12] | Doneva2017_BH, Doneva2017_NS | candidate |
| 2.1 | the general ESTGB action, V = 0, lambda of dimension length, f(phi) | [2, 13] | Doneva2017_BH | candidate |
| 2.1 | the Gauss-Bonnet invariant | [2, 14] | -- | missing: a Gauss-Bonnet invariant reference |
| 2.1 | the field equations and Gamma_munu, Psi_mu | [14, 15] | Doneva2017_BH | candidate |
| 2.1 | matter stress-energy and its conservation (Bianchi identities) | [15, 16] | -- | missing: a GR textbook or the NS-in-ESTGB derivation |
| 2.2 | conditions on f(phi) at phi = 0 so GR solves the equations | [17, 18] | Doneva2017_BH | candidate |
| 2.2 | GR solutions with phi = 0 are exact solutions | [17, 19] | Doneva2017_BH, Doneva2017_NS | candidate |
| 2.2 | linear scalar perturbation on a GR background | [20, 21] | -- | missing: the scalarization stability papers |
| 2.2 | Schrodinger-like form, negative potential well | [22, 23] | -- | missing: the effective-potential analysis |
| 2.2 | tachyonic instability, omega^2 < 0 | [24, 25] | DamourEspositoFarese1993 | candidate |
| 2.2 | bifurcation onto the scalarized branch | [1, 5] | Doneva2017_BH, Doneva2017_NS | matched |
| 2.3 | static spherically symmetric metric, Schwarzschild coordinates | [26] | -- | missing: a GR textbook |
| 2.3 | the BH coupling f(phi) = (1 - exp(-6 phi^2)) / 12 | [27] | Doneva2017_BH | matched |
| 2.3 | horizon conditions | [28] | Doneva2017_BH | candidate |
| 2.3 | horizon regularity of (dphi/dr)_H and the r_h existence bound | [29] | Doneva2017_BH | candidate |
| 2.3 | asymptotics defining the ADM mass M and the dilaton charge D | [8] | Doneva2017_BH | candidate |
| 2.4 | perfect fluid and hydrostatic equilibrium | [16] | Tolman1939, OppenheimerVolkoff1939 | candidate |
| 2.4 | the NS coupling f(phi) = (exp(-beta phi^2) - 1) / (2 beta) | [30] | Doneva2017_NS | matched |
| 2.4 | central regularity conditions | [9, 31] | Doneva2017_NS | candidate |
| 2.4 | the algebraic constraint on Lambda_2 | [19] | Doneva2017_NS | candidate |
| 2.4 | shooting to infinity yields M and D | [9, 32] | Doneva2017_NS | candidate |
| 2.4 | equations of state behind the NS dataset | -- | -- | missing: the EOS the dataset uses (Read2009 and others are in refs.bib, which one applies is unknown) |
| 2.4 | truncation past M_max (dM/drho_c < 0) and of non-converged runs | [33, 34] | -- | missing: a stability-at-M_max reference |
| 3.2 | H2: Huber loss with GELU against saturation | [35, 36] | Huber1964, Hendrycks2016_GELU | matched |
| 3.2 | H3: identity skip connections | [35, 37] | He2016_ResNet | matched |
| 4.1 | z-score standardization (StandardScaler) | [38, 39] | ScikitLearn2011 | matched |
| 4.1 | Gaussian process regression with Matern kernels | [40] | Rasmussen2005_GP | matched |
| 4.1 | gradient boosted trees (XGBoost) | -- | -- | missing: Chen and Guestrin 2016 |
| 4.1 | multilayer perceptrons | [41, 42] | -- | missing: a deep learning reference |
| 4.1 | residual networks | [43, 44] | He2016_ResNet | matched |
| 4.1 | the implementation | -- | PyTorch2019, ScikitLearn2011 | matched |
| 4.2 | Huber loss for robustness near boundaries | [36] | Huber1964 | matched |
| 4.2 | AdamW with cosine annealing | [36] | Loshchilov2017_AdamW, Kingma2014_Adam | matched |
| 4.2 | cosine annealing schedule | -- | -- | missing: Loshchilov and Hutter 2016 (SGDR) |
| 4.2 | L-BFGS fine-tuning to loss floors of 1e-8 to 1e-10 | [41] | Michaud2023_PrecisionML | matched |
| 4.3 | ensemble bagging for epistemic uncertainty | [45] | Breiman1996_Bagging | matched |
| 5.3 | parity and MARE comparison of the model pool | [46, 47, 48] | Liodis2024, Kaushikk2026, Liu2026 | candidate |
| 5.3 | the GPR O(N^3) wall | [40] | Rasmussen2005_GP | matched |
| 5.4 | speedups of 1e2 to 1e4 over ODE integration | [49, 50, 51] | Liodis2024, Kaushikk2026, Reed2024 | candidate |
| 6 | mock MCMC recovery of (beta, lambda) | [10, 52] | Reed2024 | candidate |
| 6 | the MCMC sampler | -- | -- | missing: the sampler used (emcee: Foreman-Mackey et al. 2013) |
| 6 | surrogate posterior against the solver posterior | [53] | -- | missing: a posterior-validation reference |
| 7 | future extension to rapidly rotating compact objects | -- | Komatsu1989, Cook1994, Stergioulas1995 | candidate |

refs.bib keys no site names yet: Hinderer2008, FlanaganHinderer2008, DamourNagar2009,
BinningtonPoisson2009 (tidal deformability), Abbott2017_GW170817, Abbott2020_GW190814,
Miller2019_NICER, Riley2019_NICER, Miller2021_NICER (observations), Charmousis2022, Read2009,
Lindblom2010, Margueron2018a, Tews2018a, BPS1971 (equations of state), Quarteroni2015_RBM,
Oord2016_CausalCNN, TensorFlow2015, SciPy2020, F2PY_Peterson2009.
