# Mathematical Formulation

Every algorithm in the signal path, stated formally. Notation is consistent
throughout:

| Symbol | Meaning |
|---|---|
| $\mathbf{X} \in \mathbb{R}^{C \times T}$ | one trial: $C$ channels × $T$ samples |
| $C = 8$ | channels |
| $T = 750$ | samples (3.0 s at $f_s = 250$ Hz) |
| $\mathbf{x}_c \in \mathbb{R}^{T}$ | row $c$ of $\mathbf{X}$ |
| $N$ | number of trials |
| $y \in \{0, 1\}$ | class label (left / right imagery) |

---

## 1 · Filtering

### 1.1 Butterworth response

The band-pass is a 4th-order Butterworth, whose squared magnitude response is

$$|H(j\omega)|^2 = \frac{1}{1 + \left(\dfrac{\omega}{\omega_c}\right)^{2n}}$$

with order $n = 4$ and cutoff $\omega_c$. Butterworth is chosen because it is
maximally flat in the passband — no ripple to distort the band-power estimate
that the whole method depends on.

### 1.2 Second-order sections

The filter is applied as cascaded biquads for numerical stability:

$$H(z) = \prod_{k=1}^{K} \frac{b_{0k} + b_{1k}z^{-1} + b_{2k}z^{-2}}{1 + a_{1k}z^{-1} + a_{2k}z^{-2}}$$

### 1.3 Zero-phase filtering

Applied forward then backward (`sosfiltfilt`):

$$y[n] = \mathcal{F}^{-1}\Big\{ |H(\omega)|^2 \, \mathcal{F}\{x[n]\} \Big\}$$

The effective response is $|H(\omega)|^2$ — **real-valued**, so group delay is
exactly zero:

$$\tau_g(\omega) = -\frac{d\,\angle H(\omega)}{d\omega} = 0$$

This matters because ERD timing is informative; a phase shift would smear the
onset.

**Bands used:**

| Purpose | Band |
|---|---|
| Artifact (blink) detection | 0.5 – 8 Hz, order 2 |
| Motor imagery | 8 – 30 Hz, order 4 |
| FBCSP sub-bands | 4–8, 8–12, 12–16, 16–20, 20–26, 26–32 Hz |

---

## 2 · Artifact rejection

For frontal channels $\mathcal{A} = \{\text{Fp1}, \text{Fp2}\}$, with
$\tilde{\mathbf{x}}_c$ the 0.5–8 Hz filtered signal:

$$\text{blink} = \begin{cases}
1 & \text{if } \max_{c \in \mathcal{A}} \max_{t} |\tilde{x}_c[t]| > \theta_a \\
0 & \text{otherwise}
\end{cases}
\qquad \theta_a = 100\ \mu\text{V}$$

**The filter band matters.** Blinks are 0.5–3 Hz. Applying the detector after the
8–30 Hz motor filter removes the very signal being detected — measured result: 0
of 32 blinks found.

---

## 3 · Covariance estimation

### 3.1 Sample covariance

$$\mathbf{C} = \frac{1}{T-1}\,\mathbf{X}_{0}\mathbf{X}_{0}^{\mathsf{T}} \in \mathbb{R}^{C \times C},
\qquad \mathbf{X}_0 = \mathbf{X} - \boldsymbol{\mu}\mathbf{1}^{\mathsf{T}}$$

### 3.2 Trace normalisation

$$\hat{\mathbf{C}} = \frac{\mathbf{C}}{\operatorname{tr}(\mathbf{C})}$$

Removes the overall amplitude, which varies with electrode impedance and is not
informative — only the *spatial distribution* of power matters.

### 3.3 Shrinkage regularisation

$$\mathbf{C}_{\text{reg}} = (1-\lambda)\,\hat{\mathbf{C}} + \frac{\lambda}{C}\operatorname{tr}(\hat{\mathbf{C}})\,\mathbf{I},
\qquad \lambda = 0.05$$

With $T \gg C$ this is mild, but it guarantees $\mathbf{C}_{\text{reg}} \succ 0$
(strictly positive definite), which the Riemannian methods require.

---

## 4 · Common Spatial Patterns (CSP)

### 4.1 Objective

Find a spatial filter $\mathbf{w} \in \mathbb{R}^{C}$ maximising the variance
ratio between classes:

$$J(\mathbf{w}) = \frac{\mathbf{w}^{\mathsf{T}}\mathbf{C}_0\mathbf{w}}{\mathbf{w}^{\mathsf{T}}\mathbf{C}_1\mathbf{w}}$$

This is a Rayleigh quotient. Maximising it is exactly matched to ERD: one class
has *reduced* power over one hemisphere, so the ratio is the natural statistic.

### 4.2 Solution

Stationary points satisfy the generalised eigenvalue problem

$$\mathbf{C}_0\mathbf{w} = \lambda\,(\mathbf{C}_0 + \mathbf{C}_1)\,\mathbf{w}$$

Solving gives eigenpairs $\{(\lambda_i, \mathbf{w}_i)\}_{i=1}^{C}$ with
$\lambda_i \in [0,1]$. Filters with $\lambda \to 1$ maximise class-0 variance;
$\lambda \to 0$ maximise class-1.

### 4.3 Filter selection

Take the $m/2$ largest and $m/2$ smallest eigenvectors ($m = 4$):

$$\mathbf{W} = [\mathbf{w}_1, \ldots, \mathbf{w}_{m/2}, \mathbf{w}_{C-m/2+1}, \ldots, \mathbf{w}_C]^{\mathsf{T}} \in \mathbb{R}^{m \times C}$$

### 4.4 Projection and features

$$\mathbf{Z} = \mathbf{W}\mathbf{X} \in \mathbb{R}^{m \times T}$$

$$f_i = \log\!\left(\frac{\operatorname{var}(\mathbf{z}_i)}{\sum_{j=1}^{m}\operatorname{var}(\mathbf{z}_j)}\right), \qquad i = 1 \ldots m$$

The $\log$ makes the distribution approximately Gaussian, which is what LDA
assumes. Normalising by the total makes the feature scale-invariant.

---

## 5 · Filter Bank CSP (FBCSP)

For each sub-band $b = 1 \ldots B$ ($B = 6$):

$$\mathbf{X}^{(b)} = h_b * \mathbf{X} \;\longrightarrow\; \mathbf{W}^{(b)} \;\longrightarrow\; \mathbf{f}^{(b)} \in \mathbb{R}^{m}$$

Concatenate:

$$\mathbf{f} = \big[\mathbf{f}^{(1)}; \ldots; \mathbf{f}^{(B)}\big] \in \mathbb{R}^{Bm} = \mathbb{R}^{24}$$

### 5.1 Mutual information feature selection

Select the $k = 12$ features maximising

$$I(f_i; y) = \sum_{y}\int p(f_i, y)\,\log\frac{p(f_i, y)}{p(f_i)\,p(y)}\;df_i$$

**Why a filter bank:** the individual mu peak varies between roughly 9 and 13 Hz
across people. A single fixed band is mistuned for most users; MI selects the
bands that carry information *for this person*.

---

## 6 · Riemannian geometry

Covariance matrices live on $\mathcal{M}$, the manifold of symmetric positive
definite matrices — a curved space, not $\mathbb{R}^{n}$. Treating them as flat
vectors discards that structure.

### 6.1 Affine-invariant distance

$$\delta_R(\mathbf{C}_1, \mathbf{C}_2) = \left\| \log\!\left(\mathbf{C}_1^{-1/2}\mathbf{C}_2\mathbf{C}_1^{-1/2}\right) \right\|_F
= \left(\sum_{i=1}^{C}\log^2\lambda_i\right)^{1/2}$$

where $\lambda_i$ are the eigenvalues of $\mathbf{C}_1^{-1}\mathbf{C}_2$.

This metric is invariant to any invertible linear transform
$\mathbf{C} \mapsto \mathbf{A}\mathbf{C}\mathbf{A}^{\mathsf{T}}$ — which means it
is unaffected by electrode gain, re-referencing, or volume conduction mixing.

### 6.2 Fréchet (geometric) mean

$$\mathfrak{M} = \arg\min_{\mathbf{M} \in \mathcal{M}} \sum_{i=1}^{N} \delta_R^2(\mathbf{M}, \mathbf{C}_i)$$

No closed form; solved by gradient descent

$$\mathbf{M}_{k+1} = \mathbf{M}_k^{1/2}\exp\!\left(\frac{1}{N}\sum_{i=1}^{N}\log\!\left(\mathbf{M}_k^{-1/2}\mathbf{C}_i\mathbf{M}_k^{-1/2}\right)\right)\mathbf{M}_k^{1/2}$$

### 6.3 Tangent space projection

Map each covariance to the tangent plane at $\mathfrak{M}$:

$$\mathbf{S}_i = \log\!\left(\mathfrak{M}^{-1/2}\,\mathbf{C}_i\,\mathfrak{M}^{-1/2}\right) \in \text{Sym}(C)$$

Vectorise the upper triangle, weighting off-diagonals by $\sqrt{2}$ to preserve
the Frobenius inner product:

$$\operatorname{vec}(\mathbf{S}) = \big[s_{11},\, \sqrt{2}s_{12},\, \ldots,\, \sqrt{2}s_{1C},\, s_{22},\, \ldots,\, s_{CC}\big]$$

Dimension:

$$d = \frac{C(C+1)}{2} = \frac{8 \times 9}{2} = 36$$

The tangent space is Euclidean, so any standard classifier applies — while the
manifold structure has been respected in getting there.

---

## 7 · Classifiers

### 7.1 Linear Discriminant Analysis with shrinkage

Decision function:

$$g(\mathbf{f}) = \mathbf{w}^{\mathsf{T}}\mathbf{f} + b, \qquad
\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$$

$$b = -\tfrac{1}{2}(\boldsymbol{\mu}_1 + \boldsymbol{\mu}_0)^{\mathsf{T}}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \log\frac{P(y{=}1)}{P(y{=}0)}$$

Ledoit–Wolf shrinkage:

$$\boldsymbol{\Sigma}_{\text{LW}} = (1-\gamma)\,\hat{\boldsymbol{\Sigma}} + \gamma\,\frac{\operatorname{tr}(\hat{\boldsymbol{\Sigma}})}{d}\,\mathbf{I}$$

with $\gamma^{\star}$ chosen analytically to minimise expected squared error.
Essential here because $N$ (trials) is small relative to $d$ (features).

### 7.2 Linear SVM

$$\min_{\mathbf{w}, b, \xi} \; \tfrac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{N}\xi_i
\quad \text{s.t.} \quad y_i(\mathbf{w}^{\mathsf{T}}\mathbf{f}_i + b) \ge 1 - \xi_i,\; \xi_i \ge 0$$

### 7.3 Logistic regression

$$P(y{=}1 \mid \mathbf{f}) = \sigma(\mathbf{w}^{\mathsf{T}}\mathbf{f} + b), \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

---

## 8 · SSVEP path (retained for reference)

### 8.1 Canonical Correlation Analysis

Given EEG $\mathbf{X} \in \mathbb{R}^{C \times T}$ and reference
$\mathbf{Y}_f \in \mathbb{R}^{2H \times T}$:

$$\mathbf{Y}_f = \begin{bmatrix}
\sin(2\pi f t) \\ \cos(2\pi f t) \\ \vdots \\ \sin(2\pi H f t) \\ \cos(2\pi H f t)
\end{bmatrix}, \qquad H = 4$$

Maximise the correlation of linear combinations:

$$\rho(f) = \max_{\mathbf{u}, \mathbf{v}}
\frac{\mathbf{u}^{\mathsf{T}}\mathbf{C}_{XY}\mathbf{v}}
{\sqrt{\mathbf{u}^{\mathsf{T}}\mathbf{C}_{XX}\mathbf{u}}\;\sqrt{\mathbf{v}^{\mathsf{T}}\mathbf{C}_{YY}\mathbf{v}}}$$

Both sine and cosine are included so the result is phase-invariant — the brain's
response phase is unknown.

### 8.2 Filter Bank CCA

$$\text{score}(f) = \sum_{b=1}^{B} w_b \cdot \rho_b^2(f), \qquad w_b \in \{1.0,\, 0.65,\, 0.45\}$$

$$\hat{y} = \arg\max_{f} \; \text{score}(f)$$

### 8.3 Frame quantisation

On a display of refresh rate $R$, only these frequencies are exactly renderable:

$$f_{\text{actual}} = \frac{R}{P}, \qquad P = \text{round}\!\left(\frac{R}{f_{\text{target}}}\right) \in \mathbb{Z}^{+}$$

At $R = 60$: $P = 4 \Rightarrow 15$ Hz, $P = 3 \Rightarrow 20$ Hz. Requesting
12 Hz yields $P = 5 \Rightarrow 12$ Hz exactly, but requesting 18 Hz yields
$P = 3 \Rightarrow 20$ Hz — a silent 2 Hz error.

---

## 9 · Decision logic

### 9.1 Confidence threshold

$$\hat{y} = \begin{cases}
\arg\max_k s_k & \text{if } \max_k s_k \ge \theta \\
\texttt{NO\_ACTION} & \text{otherwise}
\end{cases}$$

### 9.2 Per-user threshold from rest data

$$\theta = Q_{0.95}\big(\{\max_k s_k(\mathbf{X}_i) : \mathbf{X}_i \in \mathcal{D}_{\text{rest}}\}\big)$$

The 95th percentile of the score distribution while resting. By construction
about 5% of rest windows exceed it, bounding the false-positive rate.

### 9.3 Consecutive agreement

Emit a command at window $n$ only if

$$\hat{y}_{n} = \hat{y}_{n-1} = \hat{y}_{n-2} \neq \texttt{NO\_ACTION}$$

Under an i.i.d. false-positive assumption with per-window rate $p$, the
probability of three spurious agreements is $p^3 / K^2$ for $K$ classes — for
$p = 0.05,\, K = 2$ that is $3 \times 10^{-5}$ per window.

### 9.4 Refractory period

$$t_{\text{now}} - t_{\text{last}} \ge 1200\ \text{ms}$$

---

## 10 · Evaluation metrics

### 10.1 Cohen's d

$$d = \frac{\bar{x}_1 - \bar{x}_0}{s_p}, \qquad
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_0-1)s_0^2}{n_1 + n_0 - 2}}$$

### 10.2 Mahalanobis distance (multivariate separability)

$$D_M = \sqrt{(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^{\mathsf{T}}\,\boldsymbol{\Sigma}_p^{-1}\,(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)}$$

### 10.3 Wolpaw Information Transfer Rate

$$B = \log_2 K + P\log_2 P + (1-P)\log_2\!\left(\frac{1-P}{K-1}\right)$$

$$\text{ITR} = B \times \frac{60}{T_{\text{sel}}} \quad \text{[bits/min]}$$

with $K$ classes, accuracy $P$, and $T_{\text{sel}}$ seconds per selection.

### 10.4 Binomial test against chance

$$p\text{-value} = \sum_{i=k}^{n}\binom{n}{i}\left(\frac{1}{K}\right)^{i}\left(1-\frac{1}{K}\right)^{n-i}$$

### 10.5 Permutation test

For $M$ label shuffles $\pi_m$, retraining the full pipeline each time:

$$p = \frac{1 + \left|\{m : a(\pi_m) \ge a_{\text{obs}}\}\right|}{M + 1}$$

The strictest available test — it accounts for every source of optimism in the
pipeline, including feature selection.

### 10.6 Compound task success

For a selection requiring $n$ independent correct decisions at per-step accuracy
$P$:

$$P_{\text{task}} = P^{\,n}$$

At $P = 0.614$ and $n = 4$: $P_{\text{task}} = 0.142$. Expected attempts
$= 1/P_{\text{task}} \approx 7$.

**This equation is why per-step accuracy alone is misleading** — 61% sounds
acceptable until compounded over a realistic interaction.

---

## 11 · Windowing

$$\mathbf{X}^{(n)} = \mathbf{s}\big[\,nH : nH + W\,\big], \qquad W = 750,\; H = 250$$

Overlap fraction:

$$\eta = 1 - \frac{H}{W} = 1 - \frac{250}{750} = 0.667$$

**Consequence for evaluation:** adjacent windows share 67% of their samples, so a
random train/test split places near-duplicates on both sides. Grouped
cross-validation is therefore mandatory — measured inflation from ignoring this
was +11.0 percentage points.
