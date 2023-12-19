---
title: 'PyProximal - scalable convex optimization in Python'
tags:
  - Python
  - convex optimization
  - proximal
authors:
  - name: Matteo Ravasi
    orcid: 0000-0003-0020-2721
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
  - name: Earth Science and Engineering, Physical Sciences and Engineering (PSE), King Abdullah University of Science and Technology (KAUST), Thuwal, 23955-6900, Kingdom of Saudi Arabia
    index: 1
date: 19 December 2023
bibliography: paper.bib
---

# Summary

A broad class of problems in scientific disciplines ranging from image processing and astrophysics, 
to geophysics and medical imaging call for the optimization of convex, non-smooth objective functions. 
Whereas practitioners are usually very familiar with gradient-based optimization algorithms, commonly used 
to solve unconstrained, smooth optimization problems, proximal algorithms can be viewed as analogous tools for 
non-smooth and possibly constrained versions of such problems. PyProximal is a Python-based library aimed at 
democratize the application of convex optimization to scientific problems, providing all the required 
building blocks (i.e., proximal operators and algorithms) to define and solve complex, convex objective functions
in a high-level, abstract fashion, shielding users away from any unneeded mathematical and implementation details.


# Statement of need

`PyProximal` is a NUMFOCUS-affiliated Python package for convex optimization developed as an integral part of the `PyLops` framework. 
It provides practitioners in a variety of scientific disciplines with an easy-to-use Python based framework to 
define and solve composite convex objective functions arising in many modern inverse problems. Its API was
designed to provide a class-based and user-friendly interface to proximal operators coupled with function-based 
optimizers; because of its modular design, researchers in the field of convex optimization can also benefit from this 
package when developing new algorithms in a number of ways: first, they can easily include their newly developed proximal operators 
and solvers; second, they can compare these methods with state-of-the-art algorithms already provided in the package

`PyProximal` heavily relies on and seamlessly integrates with `PyLops` [@Ravasi:2020], our main package for matrix-free linear algebra 
and optimization. More specifically, some of `PyLops`'s linear operators and solvers used leveraged in the implementation 
of proximal operators that require access to linear operators (e.g., numerical derivatives) and least-squares solvers 
(e.g., conjugate gradient). Whilst other similar packages exist in the Python ecosystem, their design usually leads to a 
tight coupling between linear operators and solvers and proximal operators and solvers. On the other hand, by following the 
Separation of Concerns (SoC) design principle, we reduce to a minimum the overlap between the two libraries, easing both 
their development and maintenance, as well as allowing newcomers to learn how to solve inverse problems in a step-by-step manner. 
As such, `PyProximal` can be ultimately described as a light-weight extension of `PyLops` that users of the former can very easily 
learn and use in short time and with minimal additional effort.


# Mathematical framework

Convex optimization is routinely used to solve problems of the form [@Parikh:2013]:

\begin{equation}
\label{eq:problem}
\min_\mathbf{x} f(\mathbf{x}) +g(\mathbf{Lx})
\end{equation}

where $f$ and $g$ are possibly non-smooth convex functionals and $\mathbf{A}$ is a linear operator. A special case, 
appearing in many scientific applications, is represented by $f=1/2 \Vert y - \mathcal{A}(\mathbf{x})\Vert_2^2$, which identifies 
the so-called data misfit term. Here, $\mathcal{A}$ is a (possibly non-linear) modeling operator representing the underlying physical 
process the links the unknown model vector $\mathbf{x}$ to the vector of observations $\mathbf{y}$. In this case, 
we usually refer to $g$ as the regularization term, where one or multiple terms are added to the objective function to 
promote certain features in the sought after solution and/or constraint the optimization process to produce a solution
within a given set of allowed vectors.

Independent on the algorithm used to optimize such a cost function, a common feature of all proximal algorithms is
represented by the fact that one must be able to repeatedly evaluate the proximal operator of $f$ and/or $g$. The proximal 
operator of a function $f$ is defined as

\begin{equation}
\label{eq:prox}
prox_{\tau f} (\mathbf{x}) = \min_{\mathbf{y}} f(\mathbf{y}) +
        \frac{1}{2 \tau}||\mathbf{y} - \mathbf{x}||^2_2
\end{equation}

Whilst evaluating a proximal operator does itself require solving an optimization problem, these subproblems often 
admit closed form solutions or can be solved very quickly with ad-hoc specialized methods. Several of such proximal 
operators are efficiently implemented in the ``PyProximal`` library.

Finally, there exists three main families of proximal algorithms that can be used to solve various flavors of equation 
\autoref{eq:problem}, namely:

- Proximal Gradient [@Combettes:2011]: this method, also commonly referred to as the Forward-Backward Splitting (FBS)
  algorithm, is usually the preferred choice when $\mathbf{L}=\mathbf{I}$ (i.e. identity operator). Accelerated versions such 
  as the FISTA and TwIST algorithms exist and are usually preferred to the vanilla FBS method;
- Alternating Direction Method of Multipliers [@Boyd:2011]: this method is based on the well-known splitting strategy and can be used 
  for a broader class of problem than FBS and its accelerated versions.
- Primal-Dual [@Chambolle:2011]: another popular algorithm able to tackle the generic problem in equation 1 with any choice of A. 
  It reformulates the original problem into its primal-dual version of solves a saddle optimization problem.

``PyProximal`` provides implementations for these three families of algorithms; moreover, our solvers include additional features 
such as back-tracking for automatic selection of step-sizes, logging of cost function evolution, and custom callbacks.


# Code structure

``PyProximal`` aims to provide a modular, and easy-to-use Application Programming Interface (API) that scientists 
can use to define and solve convex objective functions by means of proximal algorithms. The API is composed of 
two main units as shown in Fig. 1. 

The first unit contains the entire suite of proximal operators, which are class-based objects that subclass 
the ``pylops.ProxOperator`` parent class. For each of these operators, the solution to the proximal optimization 
problem in equation \autoref{eq:prox} (and/or the dual proximal problem) is implemented in the ``prox`` 
(and/or ``dualprox``) method. As in most cases a closed-form solution exists for such a problem, our
implementation provides users with the most efficient way to evaluate a proximal operator. The second unit comprises
of so-called proximal solvers, optimization algorithms that are suited to solve problems of the form in equation \autoref{eq:problem}.

![Schematic representation of the ``PyProximal`` API.](figs/software.png){ width=90% }

# Representative PyProximal Use Cases

PyProximal has already been featured in a number of scientific publications:

- Joint inversion and segmentation of subsurface models: when inverting geophysical data for subsurface priorities, 
  a prior information that we would like to include in the inversion process is represented by the presence of a 
  discrete number of rock units, which can be parametrized in terms of their expected mean (or most likely value).  
  Ravasi and Birnie (2022) framed such a problem as a joint inversion and segmentation where the underlying optimization 
  is solved in alternating fashion using the Primal-dual algorithm.
- Plug-and-Play (PnP) priors: introduced in 2013 by Vent, in the PnP framework any proximal operator is interpreted 
  as a denoising problem and solved by means of any statistical or deep learning based denoiser. Recently, Romero


# References