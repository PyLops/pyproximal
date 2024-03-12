---
title: 'PyProximal - scalable convex optimization in Python'
tags:
  - Python
  - convex optimization
  - proximal
authors:
  - name: Matteo Ravasi
    orcid: 0000-0003-0020-2721
    corresponding: true
    affiliation: 1 
  - name: Marcus Valtonen Örnhag
    orcid: 0000-0001-8687-227X
    affiliation: 2
  - name: Nick Luiken
    orcid: 0000-0003-3307-1748
    affiliation: 1 
  - name: Olivier Leblanc
    orcid: 0000-0003-3641-1875
    affiliation: 3
  - name: Eneko Uruñuela
    orcid: 0000-0001-6849-9088
    affiliation: 4
affiliations:
  - name: Earth Science and Engineering, Physical Sciences and Engineering (PSE), King Abdullah University of Science and Technology (KAUST), Thuwal, Kingdom of Saudi Arabia
    index: 1 
  - name: Ericsson Research, Lund, Sweden.
    index: 2
  - name: ISPGroup, INMA/ICTEAM, UCLouvain, Louvain-la-Neuve, Belgium.
    index: 3
  - name: Basque Center on Cognition, Brain and Language (BCBL), Donostia-San Sebastián, Spain.
    index: 4
date: 19 December 2023
bibliography: paper.bib
---


# Summary

A broad class of problems in scientific disciplines ranging from image processing and astrophysics, 
to geophysics and medical imaging call for the optimization of convex, non-smooth objective functions. 
Whereas practitioners are usually familiar with gradient-based algorithms, commonly used 
to solve unconstrained, smooth optimization problems, proximal algorithms can be viewed as analogous tools for 
non-smooth and possibly constrained versions of such problems. These
algorithms sit at a higher level of abstraction than gradient-based algorithms and 
require a basic operation to be performed at each iteration: the evaluation of the so-called proximal operator of the
functional to be optimized. ``PyProximal`` is a Python-based library aimed at 
democratizing the application of convex optimization to scientific problems; it provides the required 
building blocks (i.e., proximal operators and algorithms) to define and solve complex, convex objective functions
in a high-level, abstract fashion, shielding users away from any unneeded mathematical and implementation details.


# Statement of need

`PyProximal` is a Python library for convex optimization, developed as an integral part of the `PyLops` framework. 
It provides practitioners with an easy-to-use framework to define and solve composite convex objective functions 
arising in many modern inverse problems. Its API is designed to offer a class-based and user-friendly interface 
to proximal operators, coupled with function-based optimizers; because of its modular design, researchers in the field 
of convex optimization can also benefit from this library in a number of ways when developing new algorithms: first, 
they can easily include their newly developed proximal operators and solvers; second, they can compare these methods 
with state-of-the-art algorithms already provided in the library.

Several projects in the Python ecosystem provide implementations of proximal operators and/or algorithms, 
which present some overlap with those available in `PyProximal`. A (possibly not exhaustive) list of other projects is composed of
*proxalgs* [@Maheswaranathan], *proxmin* [@Melchior], *The Proximity Operator Repository* [@Chierchia], *ProxImaL* [@Heide:2016], 
and *pyxu* [@pyxu-framework]. A key common feature of all of the above mentioned packages is to be self-contained; as such, not only proximal operators and solvers
are provided, but also linear operators that are useful for the applications that the package targets. Moreover, to the best of our knowledge, all of these packages
provide purely CPU-based implementations (apart from *pyxu*). On the other hand, `PyProximal` heavily relies on and seamlessly integrates with `PyLops` [@Ravasi:2020], a Python library for matrix-free linear algebra 
and optimization. As such, it can easily handle problems with millions of unknowns and inherits 
the interchangle CPU/GPU backend of PyLops [@Ravasi:2021]. More specifically, `PyLops` is leveraged in the implementation of proximal operators that require 
access to linear operators (e.g., numerical derivatives) and/or least-squares solvers 
(e.g., conjugate gradient). Whilst libraries with similar capabilities exist in the Python ecosystem, their design usually leads to a 
tight coupling between linear and proximal operators, and their respective solvers. On the other hand, by following the 
Separation of Concerns (SoC) design principle, the overlap between `PyLops` and `PyProximal` is reduced to a minimum, easing both 
their development and maintenance, as well as allowing newcomers to learn how to solve inverse problems in a step-by-step fashion. 
As such, `PyProximal` can be ultimately described as a light-weight extension of `PyLops` that users of the latter can easily 
learn and adopt with minimal additional effort.


# Mathematical framework

Convex optimization is routinely used to solve problems of the form [@Parikh:2013]:

\begin{equation}
\label{eq:problem}
\min_\mathbf{x} f(\mathbf{x}) +g(\mathbf{Lx})
\end{equation}

where $f$ and $g$ are possibly non-smooth convex functionals and $\mathbf{L}$ is a linear operator. A special case 
appearing in many scientific applications is represented by $f=1/2 \Vert \mathbf{y} - \mathcal{A}(\mathbf{x})\Vert_2^2$. 
Here, $\mathcal{A}$ is a (possibly non-linear) modeling operator, describing the underlying physical 
process that links the unknown model vector $\mathbf{x}$ to the vector of observations $\mathbf{y}$. In this case, 
we usually refer to $g$ as the regularizer, where one or multiple functions are added to the data misfit term to 
promote certain features in the sought after solution and/or constraint the solution to fall within a given set of allowed vectors.

A common feature of all proximal algorithms is represented by the fact that one must be able to repeatedly 
evaluate the proximal operator of $f$ and/or $g$. The proximal operator of a function $f$ is defined as

\begin{equation}
\label{eq:prox}
prox_{\tau f} (\mathbf{x}) = \min_{\mathbf{y}} f(\mathbf{y}) +
        \frac{1}{2 \tau}||\mathbf{y} - \mathbf{x}||^2_2
\end{equation}

Whilst evaluating a proximal operator does itself require solving an optimization problem, these problems often 
admit closed form solutions or can be solved very efficiently with ad-hoc specialized methods. Several of such proximal 
operators are efficiently implemented in the ``PyProximal`` library.

Finally, there exists three main families of proximal algorithms that can be used to solve various flavors of 
\autoref{eq:problem}, namely:

- Proximal Gradient [@Combettes:2011]: this method, also commonly referred to as the Forward-Backward Splitting (FBS)
  algorithm, is usually the preferred choice when $\mathbf{L}=\mathbf{I}$ (i.e. identity operator). Accelerated versions such 
  as the FISTA and TwIST algorithms exist and are usually preferred to the vanilla FBS method;
- Alternating Direction Method of Multipliers [@Boyd:2011]: this method is based on a splitting strategy and can be used 
  for a broader class of problem than FBS and its accelerated versions.
- Primal-Dual [@Chambolle:2011]: another popular algorithm able to tackle problems in the form of \autoref{eq:problem} with any choice of $\mathbf{L}$. 
  It reformulates the original problem into its primal-dual version and solves a saddle optimization problem.

``PyProximal`` provides implementations for these three families of algorithms; moreover, all solvers include additional features 
such as back-tracking for automatic selection of step-sizes, logging of the objective function evolution through iterations, 
and possibility to inject custom callbacks.


# Code structure

``PyProximal``'s modular and easy-to-use Application Programming Interface (API) allows scientists 
to define and solve convex objective functions by means of proximal algorithms. The API is composed of 
two main part as shown in Fig. 1. 

The first part contains the entire suite of proximal operators, which are class-based objects subclassing 
the ``pylops.ProxOperator`` parent class. For each of these operators, the solution to the proximal optimization 
problem in \autoref{eq:prox} (and/or the dual proximal problem) is implemented in the ``prox`` 
(and/or ``dualprox``) method. As in most cases a closed-form solution exists for such a problem, our
implementation provides users with the most efficient way to evaluate a proximal operator. The second part comprises
of so-called proximal solvers, optimization algorithms that are suited to solve problems of the form in \autoref{eq:problem}. 
Finally, some specialized solvers that rely on one or more of the previously described optimizers are also provided.

![Schematic representation of the ``PyProximal`` API.](figs/software.png){ width=90% }

# Representative PyProximal Use Cases

Examples of PyProximal applications in different scientific fields include:

- *Joint inversion and segmentation of subsurface models*: when inverting geophysical data for subsurface priorities, 
  prior information can be provided to inversion process in the form of discrete number of rock units; this can be 
  parametrized in terms of their expected mean (or most likely value). @Ravasi:2022 and @Romero:2023 frame such a problem 
  as a joint inversion and segmentation task, where the underlying objective function is optimized in alternating fashion 
  using the Primal-Dual algorithm.
- *Plug-and-Play (PnP) priors*: introduced in 2013 by @Venkatakrishnan:2013, the PnP framework lays its foundation on
  the interpretation of the proximal operator as a denoising problem; as such, powerful statistical or deep learning 
  based denoisers are used to evaluate the proximal operator of implicit regularizers. @Romero:2022 applies this concept
  in the context of seismic inversion, achieving results of superior quality in comparison to traditional model-based
  regularization techniques.
- *Multi-Core Fiber Lensless Imaging* (MCFLI) is a computational imaging technique to reconstruct biological 
  samples at cellular scale. Leveraging the rank-one projected interferometric sensing of the MCFLI has been shown to 
  improve the efficiency of the acquisition process [@Leblanc:2023]; this entails solving a regularized inverse problem with
  the proximal gradient method. Depending on the image to be reconstructed, the regularization term may for instance be $L_1$ or TV.

# References