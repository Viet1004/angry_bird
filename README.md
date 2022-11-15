# Angry Bird

(This project is in the context of the course MAP556 - Monte Carlo methods at the école Polytechnique, Paris, France)

## Problem description

Everyone knows about the game Angry Bird. It's about launching a bird toward a target (the pigs). Given the distance, without wind, mathematically, we can calculate exactly the force and direction for correctly landing on the target. Assume now that the wind is stochastic and we can have control over the bird on the trajectory. The objective is that given the position and the time of the bird in the air, find the best control which minimize the cost function (defined below)

Notation:
1. m: the mass of the Angry Bird. m = 1.
2. $X_t$: the position of the bird
3. $g$: gravity force $g = (0, -4)^T$ 
4. $\lambda$: air resistance
5. $\bold{V}$: the wind. It is a vector of two components
6. $u_t$: velocity control 

The dynamic of $X_t$

$$
\left\{\begin{aligned}
\dot{X}_t &:=\frac{\mathrm{d} X_t}{\mathrm{~d} t}=\dot{X}_0+\mathbf{g} t-\lambda X_t+V_t+u_t \\
X_t &=\dot{X}_0 t+\mathbf{g} \frac{t^2}{2}-\lambda \int_0^t X_s \mathrm{~d} s+\int_0^t V_s \mathrm{~d} s+\int_0^t u_s \mathrm{~d} s
\end{aligned}\right.
$$

For the simplicity of the problem, we consider the discrete model where we only have the control after each second. As a result, the discrete evolution of $X_t$ is:

$$
X_{t_{i+1}}=X_{t_i}+\dot{X}_0 \Delta_T+\mathbf{g}\left(\frac{t_{i+1}^2}{2}-\frac{t_i^2}{2}\right)-\lambda X_{t_i} \Delta_T+V_{t_i} \Delta_T+u_{t_i} \Delta_T
$$
where $\Delta_T = 0.1$

The wind $V$ is modeled by the stochastic differential equation:

$$
V_t=\left(\begin{array}{c}
V_{1, t} \\
V_{2, t}
\end{array}\right):=-\int_0^t\left(\begin{array}{c}
0.9 \times V_{1, s}+0.2 \times V_{2, s} \\
V_{1, s}+1.1 \times V_{2, s}
\end{array}\right) \mathrm{d} s+5\left(\begin{array}{l}
W_{1, t} \\
W_{2, t}
\end{array}\right)
$$

Inital conditions:
The target is at (200, 0).
The time the bird is on the air is T = 10. Since we only have the control over the bird at t = (0,1,$\ldots$,9), we have 10 controls $u_t$.

The cost function of each trajectory: 
$$
j(u):=\sum_{i=0}^9\left|u_i\right|^2+L\left(X_{10}\right)
$$

where the terminal function is:

$$
\begin{aligned}
L(x)=&\left(\frac{x_1-D-x_2}{\sqrt{2}}+\frac{\left(x_1-D-x_2\right)_{+}}{\sqrt{2}}\right)^2+\left(\frac{x_1-D+x_2}{\sqrt{2}}\right)^2 \\
&+\left(x_1+x_2-(D-15)\right)_{-}^2
\end{aligned}
$$

The empirique cost function is 
$$
J(u) \approx J^M(u):=\frac{1}{M} \sum_{m=1}^M(j(u))^{(m)}
$$

## Method:

To solve this problem, we implement the algorithm NNContPI in the paper
https://hal.archives-ouvertes.fr/hal-01949221v3/document

## Execution

In the code, we include the trained models. In the original code, we don't use GPU so the training time is very long. It will be updated the next time.

To run the code, 
```
python controle_etudiant.py
```

To train the model, uncomment the last two lines in mcc2.py