GLT
***

Where do the GLTs come from?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main aim of this paragraph is to present a crucial example that highlights the importance of the GLT algebra when dealing with linear systems coming from the discretization of PDEs. Let us start with some preliminaries. In detail, we will recall the notion of symbol of a matrix-sequence and the basic idea behind the GLT theory.


Spectral preliminaries
______________________

.. %\subsubsection{Symbol}

The following one is a rather informal definition of symbol of a matrix-sequence.

.. .. math:: 
.. 
..   \begin{definition}\label{def}Let
.. 
..   \begin{itemize}
..           \item $\{A_n\}_n$ be a matrix-sequence, ${\rm dim}(A_n)=d_n\uparrow\infty$,
..           \item $f:D\subset\mathbb R^d\to\mathbb C$,\quad $0<{\rm measure}(D)<\infty$.
..   \end{itemize}
..   We say that $\{A_n\}_n$ is \emph{spectrally distributed} as $f$ (in symbols $\{A_n\}_n\sim_\lambda f$) if the eigenvalues of $A_n$ are approximately a uniform sampling of $f$ over $D$. In turn, the function $f$ is called \emph{spectral symbol} of $\{A_n\}_n$.
..   \end{definition}

**example**:

When :math:`d_n=n`, :math:`d=1`, :math:`D=[0,\pi]`, :math:`\{A_n\}_n\sim_\lambda f` means

.. .. math::
.. 
..   \begin{equation*}
..   \lambda_j(A_n)\approx f\left(\frac{j\pi}{n}\right), \quad j=0, \ldots, n-1.
..   \end{equation*}

.. \begin{remark} This definition can also be given is the singular values sense (replacing $f\rightarrow |f|$). Notation: $\{A_n\}_n\sim_\sigma f$.
.. \end{remark}
.. 
.. In what follows, we briefly recall the basics of the GLT theory. For the sake of simplicity, we focus on the 1D setting.
.. %\subsubsection{GLT theory}
.. \begin{definition}\label{glt}
.. A \emph{GLT sequence} is a matrix-sequence $\{A_n\}_n$, with ${\rm dim}(A_n)=d_n\uparrow\infty$ obtained as \emph{linear combination, product, inversion or conjugation} of
.. \begin{itemize}
.. \item $\{T_n(f)\}_n$, \emph{Toeplitz sequences} with $T_n(f)$ $f: \left[-\pi,\pi \right] \rightarrow \mathbb{C}$, $f\in L^1([-\pi,\pi])$ defined as follows
.. \[ T_n(f) = \left[\begin{array}{cccc}
.. f_0 & f_{-1} & \cdots & f_{-(n-1)}\\
.. f_1 & \ddots & \ddots &  \vdots\\
.. %\vdots & \ddots & \ddots & \ddots & \vdots\\
.. \vdots & \ddots & \ddots & f_{-1}\\
.. f_{n-1} & \cdots  & f_1 & f_0
.. \end{array}\right],\]
.. where
.. \begin{equation*}
.. f_j = \frac1{2\pi}\int_{-\pi}^\pi f(\theta)\textup{e}^{-\textup{i}j\theta}d\theta,\qquad j\in\mathbb Z,
.. \end{equation*}
.. \item $\{D_n(a)\}_n$, \emph{diagonal sampling sequences}, with $D_n(a)$, $a: \left[0,1\right] \rightarrow \mathbb{C}$ Riemann integrable function, s.t.
.. \begin{equation} \label{dsm} D_n(a) = \left[\begin{array}{cccc}
.. a(\frac1n) & & & \\
.. & a(\frac2n) & & \\
.. & & \ddots & \\
.. & & & a(1)
.. \end{array}\right], \end{equation}
.. \item $\{Z_n\}_n$, \emph{low-rank+small-norm sequences}, with
.. $$Z_n=R_n+S_n,$$
.. and
.. \begin{itemize}
.. \item $R_n$: $\lim_{n\to\infty}\frac{\textup{rank}(R_n)}{d_n}=0$,
.. \item $S_n$: $\lim_{n\to\infty}\|S_n\|=0$.
.. \end{itemize}
.. Here and throughout these notes, $\|A_n\|:=\sigma_{\rm max}(A_n)$.
.. \end{itemize}
.. \end{definition}
.. \begin{remark}\label{rem}
.. Let us observe that according to Definition \ref{glt}
.. \begin{enumerate}
.. \item the sequences $\{T_n(f)\}_n$, $\{D_n(a)\}_n$, and $\{Z_n\}_n$ are GLT sequences themselves;
.. \item the set of the GLT sequences form a $*$-algebra, i.e., it is closed under linear combinations, products, inversion, conjugation: hence, the sequence obtained via algebraic operations on a finite set of input GLT sequences is still a GLT sequence.
.. \end{enumerate}
.. \end{remark}
.. 
.. What follows provides a link between the notions of symbol and of GLT sequence.
.. \begin{theorem}\label{szego} Let $f: \left[-\pi,\pi \right] \rightarrow \mathbb{C}$, with $f\in L^1([-\pi,\pi])$. Then
.. \begin{equation*}
.. \{T_n(f)\}_n\sim_\sigma f.
.. \end{equation*}
.. \end{theorem}
.. \begin{remark}\label{diag} Let $\{D_n(a)\}_n$ be a diagonal sampling sequence defined as in \eqref{dsm}. Then, recalling Definition \ref{def}, it yields
.. \[\{D_n(a)\}_n\sim_{\sigma,\lambda} a.\]
.. \end{remark}
.. \begin{theorem}\label{zero}
.. Let $\{Z_n\}_n$ be a matrix-sequence. Then the following conditions are equivalent
.. \begin{itemize}
.. \item $\{Z_n\}_n\sim_\sigma 0$;
.. \item For any $n$ we have $Z_n=R_n+S_n$, where $\lim_{n\to\infty}\frac{\textup{rank}(R_n)}{N_n}=\lim_{n\to\infty}\|S_n\|=0$.
.. \end{itemize}
.. \end{theorem}
.. Because of the equivalence in Theorem \ref{zero} the low-rank+small-norm matrix-sequences are also known as \emph{zero-distributed} matrix-sequences.
.. 
.. 
.. Summarizing, thanks to Theorems \ref{szego},\ref{zero} and Remark \ref{diag}, all the basic GLT sequences (Toeplitz, diagonal sampling, zero-distributed) are equipped with a symbol and then an asymptotical description of their spectrum is available. More in general, as stated in the following proposition, each GLT sequence is provided with a symbol.
.. \begin{proposition}
.. Each GLT sequence $\left\{A_n\right\}_n$ is equipped with a symbol in the singular value sense, i.e. there exists a function $\chi: [0,1]\times[-\pi,\pi]\rightarrow\mathbb{C}$ such that $$\left\{A_n\right\}_n \sim_{\sigma} \chi.$$
.. \end{proposition}
.. 
.. At this point a question arises: how to compute the symbol of a GLT sequence? The following proposition provides the answer.
.. \begin{proposition}\label{sym}
.. Let $\{ A_{n} \}_{n}$ and $\{ B_{n} \}_{n}$ be two GLT sequences such that $\{ A_{n} \}_{n}\sim_\sigma \kappa_1$ and $\{ B_{n} \}_{n}\sim_\sigma \kappa_2$. Then
..         \begin{itemize}
.. 
..         \item $\{\alpha A_{ n} + \beta B_{ n}\}_{n} \sim_\sigma \alpha\kappa_1+\beta \kappa_2, \quad \alpha, \beta \in \mathbb{C};$
..         \item $\{A_{n}B_{n}\}_{n} \sim_\sigma\kappa_1 \kappa_2;$
..         \item if $\kappa_1$ vanishes, at most, in a set of zero Lebesgue measure, then $\{A^{-1}_{n}\}_{n} \sim_\sigma \kappa_1^{-1};$
..         \item $\{ A_{n}^{H} \}_{n} \sim_\sigma\bar{\kappa_1}.$
..       \end{itemize}
.. \end{proposition}
.. 
.. Let us summarize the previous results in the following three properties:
.. \begin{itemize}
.. \item[\textbf{GLT1}] $\left\{A_n\right\}_n$ a GLT sequence $\Rightarrow$ $\left\{A_n\right\}_n \sim_{\sigma} \chi$ with $\chi: [0,1]\times[-\pi,\pi]\rightarrow\mathbb{C}$.
.. \item[\textbf{GLT2}] The GLT sequences form a $*$-algebra and the symbol of a GLT sequence is computed according to Proposition \ref{sym}.
.. \item[\textbf{GLT3}]
.. \begin{itemize}
.. \item if $f\in L^1([-\pi,\pi])$, then $\{T_n(f)\}_n$ is a GLT sequence and $\{T_n(f)\}_n\sim_\sigma f$;
.. \item if $a:[0,1]\rightarrow\mathbb{C}$ is Riemann integrable, then $\{D_n(a)\}_n$ is a GLT sequence and $\{D_n(a)\}_n\sim_{\sigma,\lambda} a$;
.. \item if $\{Z_n\}_n$ is such that for any $n$ $Z_n=R_n+S_n$, where $\lim_{n\to\infty}\frac{\textup{rank}(R_n)}{N_n}=\lim_{n\to\infty}\|S_n\|=0$, then $\{Z_n\}_n$ is a GLT sequence and $\{Z_n\}_n\sim_\sigma0$.
.. \end{itemize}
.. \end{itemize}
.. 
.. \subsection*{GLTs come into life: an FD discretization of a diffusion PDE}
.. Let us consider the following diffusion problem
.. \begin{eqnarray}\label{pro} \left\{\begin{array}{l}
.. -(a(x)u'(x))'=g(x),\qquad x\in(0,1),\vspace{5pt}\\
.. u(0)=u(1)=0,
.. \end{array}\right.
.. \end{eqnarray}
.. where $a,g\in C([0,1])$.
.. Let us discretize it with a classical second-order central FD scheme. Let $n\in\mathbb{N}$ be the discretization parameter, and set $h=\frac{1}{n+1}$ and $x_j=jh$ for all $j\in[0,n+1]$. For $j=1,\ldots,n$
.. \begin{align*} \left.-(a(x)u'(x))'\right|_{x=x_j}&\approx-\frac{a(x_{j+\frac12})u'(x_{j+\frac12})-a(x_{j-\frac12})u'(x_{j-\frac12})}{h}\\[5pt]
.. &\approx-\frac{a(x_{j+\frac12})\dfrac{u(x_{j+1})-u(x_j)}{h}-a(x_{j-\frac12})\dfrac{u(x_j)-u(x_{j-1})}{h}}{h}\\
.. &=\frac{-a(x_{j+\frac12})u(x_{j+1})+\bigl(a(x_{j+\frac12})+a(x_{j-\frac12})\bigr)u(x_j)-a(x_{j-\frac12})u(x_{j-1})}{h^2}.
.. \end{align*}
.. Let us define $u_j\approx u(x_j)$, $j=1,\ldots,n$ and $u_0=u_{n+1}=0$. Then problem \eqref{pro} can be reformulated in a matrix form as follows: find $\mathbf u_n=(u_1,\ldots,u_n)^T$ s.t.
.. \[ A_n\mathbf u_n=\mathbf g_n\]
.. where $\mathbf g_n=h^2(g(x_1),\ldots,g(x_n))^T$ and
.. \begin{eqnarray}\label{matr}
.. A_n =
.. \begin{bmatrix}
.. a(x_{\frac12})+a(x_{\frac32}) & -a(x_{\frac32}) & & & \\[12pt]
.. -a(x_{\frac32}) & a(x_{\frac32})+a(x_{\frac52}) & -a(x_{\frac52}) & & \\[12pt]
.. & \ddots & \ddots & \ddots & \\[12pt]
.. & & -a(x_{n-\frac32}) & a(x_{n-\frac32})+a(x_{n-\frac12}) & -a(x_{n-\frac12}) \\[12pt]
.. & & & -a(x_{n-\frac12}) & a(x_{n-\frac12})+a(x_{n+\frac12})
.. \end{bmatrix}.
.. \end{eqnarray}
.. \begin{remark}
.. Note that when $a(x)\equiv1$, then $A_n$ is the Laplacian matrix
.. \begin{eqnarray*}
.. A_n = \left[\begin{array}{ccccc}
.. 2 & -1 & 0 & \cdots & 0\\
.. -1 & 2 & -1 & \ddots & \vdots\\
.. 0 & \ddots & \ddots & \ddots & 0\\
.. \vdots &\ddots & -1 & 2 & -1\\
.. 0 & \cdots & 0 & -1 & 2
.. \end{array}\right],
.. \end{eqnarray*}
.. or, in other words, the Toeplitz matrix associated to the function $2-2\cos\theta$, i.e., $A_n=T_n(2-2\cos\theta)$. Using Theorem \ref{szego} it holds that $$\{A_n\}\sim_\sigma 2-2\cos\theta.$$
.. \end{remark}
.. Our aim is to recognize in the matrix-sequence $\{A_n\}_n$, with $A_n$ defined as in \eqref{matr} a GLT sequence. Consider the following matrix
.. \begin{align*}
.. B_n=D_n(a)T_n(2-2\cos\theta)&= \left[\begin{array}{cccc}
.. a(\frac1n) & & & \\
.. & a(\frac2n) & & \\
.. & & \ddots & \\
.. & & & a(1)
.. \end{array}\right]
.. \left[\begin{array}{cccc}
.. 2 & -1 & & \\
.. -1 & \ddots & \ddots & \\
..  & \ddots & \ddots & -1\\
..  & & -1 & 2
.. \end{array}\right]
.. \end{align*}
.. \begin{align*}
.. =\begin{bmatrix}
.. 2a(\frac1n)\, & -a(\frac1n) & & & \\[5pt]
.. -a(\frac2n) & 2\,a(\frac2n) & -a(\frac2n) & & \\[5pt]
.. & \ddots & \ddots & \ddots & \\[5pt]
.. & & -a(\frac{n-1}n) & 2\,a(\frac{n-1}n) & -a(\frac{n-1}n) \\[5pt]
.. & & & -a(1) & 2\,a(1)
.. \end{bmatrix}.
.. \end{align*}
.. Let us define $Z_n=A_n-B_n$. In the view of the inequalities
.. \begin{equation*}
.. \left|x_j-\frac{j}{n}\right|=\left|\frac{j}{n+1}-\frac{j}{n}\right|\le h, \quad j=1,\ldots,n,
.. \end{equation*}
.. and thanks to the continuity of $a$, the off-diagonal entries of $Z_n$ satisfy
.. \begin{align*}
.. \left|a\left(\frac{j}{n}\right)-a(x_{j+\frac12})\right|&\le \omega_a(3h/2),\\
.. \left|a\left(\frac{j}{n}\right)-a(x_{j-\frac12})\right|&\le \omega_a(3h/2),
.. \end{align*}
.. where $\omega_a$ is the modulus of continuity of $a$. As a consequence, the modulus of each diagonal entry of $Z_n$ is bounded by $2\omega_a(3h/2)$. Therefore, it holds that
.. \begin{align*}
.. \|Z_n\|_1:=\max_{j=1,\ldots,n}\sum_{i=1}^n|z_{ij}|&\le 4 \omega_a(3h/2),\\
.. \|Z_n\|_\infty:=\max_{i=1,\ldots,n}\sum_{j=1}^n|z_{ij}|&\le 4 \omega_a(3h/2),
.. \end{align*}
.. and then
.. \begin{equation*}
.. \|Z_n\|\le\sqrt{\|Z_n\|_1\|Z_n\|_\infty}\le 4 \omega_a(3h/2)\underset{h\rightarrow 0}{\longrightarrow} 0
.. \end{equation*}
.. (recall that when $a$ is a continuous function, then $\omega_a(h)\rightarrow0$ as $h\rightarrow 0$). Therefore, $\{Z_n\}_n$ is a small-norm sequence. This, together with Remark \ref{rem}, let us to conclude that $\{A_n\}_n$ with
.. $$A_n=B_n+Z_n=D_n(a)T_n(2-2\cos\theta)+Z_n.$$
.. is a GLT sequence.
.. 
.. Now, it is easy to compute the symbol of $\{A_n\}_n$. By \textbf{GLT3} we have that
.. \begin{itemize}
.. \item $\{T_n(2-2\cos\theta)\}_n \sim_\sigma 2-2\cos\theta$;
.. \item $\{D_n(a)\}_n\sim_\sigma a$;
.. \item $\{Z_n\}_n\sim_\sigma 0$;
.. \end{itemize}
.. and then by \textbf{GLT2} it holds
.. $$\{A_n\}_n\sim_\sigma a(x)\cdot (2-2\cos\theta).$$


.. rubric:: References

.. bibliography:: refs_glt.bib
   :cited:

