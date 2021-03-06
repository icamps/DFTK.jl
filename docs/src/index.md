# DFTK.jl: The density-functional toolkit.

DFTK is a `julia` package of for playing with
plane-wave density-functional theory algorithms.

TODO the notations in this file are out of date. Look at the examples or at the source code for updated documentation.

## Terminology and Definitions
The general terminology used throughout the documentation
of the plane-wave aspects of the code.

## Lattices
Usually we denote with $A$ the matrix containing
all lattice vectors as columns and with
```math
\textbf{B} = 2\pi \textbf{A}^{-T}
```
the matrix containing the reciprocal lattice vectors as columns.

## Units
Unless otherwise stated the code and documentation uses
atomic units and fractional or integer coordinates for $k$-Points
and wave vectors $G$.
The equivalent Vectors in cartesian coordiates will be denoted
as $k^c$ or $G^c$, i.e.
```math
k^c = \textbf{B} k \quad G^c = \textbf{B} G.
```

## Plane wave basis functions
At the moment the code works exclusively with orthonormal plane waves.
In other words our bases consist of functions
```math
e_{G^c} = 1/\sqrt{\Omega} e^{i\, G^c \cdot x}
```
where $\Omega$ is the unit cell volume
and $G^c$ is a wave vector in cartesian coordiates.

## Basis sets

- The **wave-function basis** $B_k$, consisting of all
  plane-wave basis functions below the desired energy cutoff $E_\text{cut}$
  for each $k$-point:
  ```math
  B_k = \{ e_{G^c} : 1/2 |G^c + k^c|^2 ≤ E_\text{cut}.
  ```
  Geometrically the corresponding wave vectors $G^c$
  form a ball of radius $\sqrt{2 E_\text{cut}}$ centred at $k^c$.
  This makes the corresponding set of $G$-vectors
  ```math
  \{ G : |\textbf{B} (G + k)| ≤ 2 \sqrt{E_\text{cut}} \}
  ```
  in integer coordinates an ellipsoid.

- The **potential** or **density basis** $B_\rho$, consisting of
  all plane waves on which a potential needs to be known in order to be
  consistent with the union of all $B_k$ for all $k$. This means that
  it is the set
  ```math
  B_\rho = \{ e_{G^c} - e_{\tilde{G}^c} : e_{G^c}, e_{\tilde{G}^c} \in B_k \}.
  ```
  This is equivalent to the alternative definition
  ```math
  B_\rho = \{ e_{G^c} : 1/2 |G^c|^2 ≤ α^2 E_\text{cut} \},
  ```
  for a supersampling factor $\alpha = 2$.
  Geometrically this is again a ball in cartesian coordinates
  and an ellipsoid in integer coordinates.

- In practice we do not use $B_\rho$ in the code, since fast-fourier transforms
  (FFT) operate on rectangular grids instead.
  For this reason the code determines $C_\rho$,
  the smallest rectangular grid in integer coordinates
  which contains all $G$-vectors corresponding to the plane waves of $B_\rho$.
  For this we take
  ```math
  C_\rho = \{ G = (G_1, G_2, G_3)^T : |G_i| ≤ N_i \}
  ```
  where the bounds $N_i$ are determined as follows.
  Since $G = \textbf{B}^{-1} G^c$ one can employ
  Cauchy-Schwartz to get
  ```math
  N_i = max_{|G^c|^2 ≤ 2 α^2 E_\text{cut}}(\textbf{B}^{-1}[i, :] \cdot G^c)
      ≤ |\textbf{B}^{-1}[i, :]| \sqrt{2 α^2 E_\text{cut}}.
  ```
  With $\textbf{B}^{-1} = frac{1}{2\pi} \textbf{A}^T$ therefore
  ```math
  N_i ≤ |\textbf{A}[:, i]| \frac{\sqrt{2 α^2 E_\text{cut}}}{2π}
  ```
  where e.g. $\textbf{A}[:, i]$ denotes the $i$-th column of $\textbf{A}$.
  Notice, that this makes $C_\rho$ is a rectangular shape in integer
  coordinates, but a parallelepiped in cartesian coordinates.

###### TODO not yet properly updated from here

- The **XC basis** $B_\text{XC}$, which is used for computing the application
  of the exchange-correlation potential operator to the density $\rho$,
  represented in the basis $B_\rho$, that is
  ```math
  B_\text{XC}  = \{e_G : 1/2 |G|_\text{max}^2 ≤ β E_\text{cut} \}.
  ```
  Since the exchange-correlation potential might involve arbitrary powers of the
  density $ρ$, a numerically exact computation of the integral
  ```math
  \langle e_G | V_\text{XC}(ρ) e_{G'} \rangle \qquad \text{with} \qquad e_G, e_{G'} ∈ B_{Ψ,k}
  ```
  requires the exchange-correlation supersampling factor $\beta$ to be infinite.
  In practice, $\beta =4$ is usually chosen, such that $B_\text{XC} = B_\rho$.

## Real-space grids
Due to the Fourier-duality of reciprocal-space and real-space lattice,
the above basis sets define corresponding real-space grids as well:

- The grid $B_\rho^\ast$, the **potential integration grid**,
  which is the grid used for convolutions of a potential with the discretized
  representation of a DFT orbital. It is simply the iFFT-dual real-space grid
  of $B_\rho$.
- The grid $B^\ast_\text{XC}$, the **exchange-correlation integration grid**,
  i.e. the grid used for convolutions of the exchange-correlation functional
  terms with the density or derivatives of it. It is the iFFT-dual of $B_\text{XC}$.
