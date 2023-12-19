---
title: 'PyProximal - scalable convex optimization in Python'
tags:
  - Python
  - optimization
  - proximal
authors:
  - name: Matteo Ravasi
    orcid: 0000-0003-0020-2721
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
  - name: Earth Science and Engineering, Physical Sciences and Engineering (PSE), King Abdullah University of Science and Technology (KAUST), Thuwal, 23955-6900, Kingdom of Saudi Arabia
    index: 1
date: 19 December 2023
bibliography: paper.bib

  

# Summary

A broad class of problems in scientific disciplines ranging from image processing and astrophysics, 
to geophysics and medical imaging call for the optimization of convex, non-smooth objective functions. 
PyProximal is a Python-based library providing users with an extensive suite of state-of-the-art proximal 
operators and algorithms.

Whereas practitioners are usually very familiar with gradient-based optimization and the associated 
first- or second-order iterative schemes commonly used to solve unconstrained, smooth optimization problems, proximal
algorithms can be viewed as analogous tools for non-smooth and possibly constrained versions of such problems. These
algorithms sit at a higher level of abstraction than classical algorithms like steepest descent or Newton’s method and 
require a basic operation to be performed at each iteration: the evaluation of the so-called proximal operator of the
functional to be optimized.

In summary, PyProximal aims to democratize the application of convex optimization to scientific problems, providing 
users with all the required building blocks (i.e., proximal operators and algorithms) to define and solve complex,
convex objective functions in a high-level, abstract fashion, shielding them away from any unneeded mathematical and 
implementation details.


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

`PyProximal` was designed to be used by both researchers and students in applied mathematics and engineering courses.
It has already been featured in a number of scientific publications [XX] and in a graduate-level course on inverse problems [XX]. 
As the adoption of the library grows across many disciplies, we believe that `PyProximal` will enable exciting scientific discoveries 
in a variety of scientific problems with societal impact. 


# Mathematical framework

Convex optimization is routinely used to solve problems of the form:

\begin{equation}
\label{eq:problem}
min_\mathbf{x} f(\mathbf{x}) +g(\mathbf{Ax})
\end{equation}

where $f$ and $g$ are possibly non-smooth convex functionals and $\mathbf{A}$ is a linear operator. A special case, 
appearing in many scientific applications, is represented by f=1/2 \Vert y - \mathbf{Gx}\Vert_2^2, which identifies 
the so-called data misfit term. Here, $\mathbf{G}$ is the modeling operator representing the underlying physical 
process the links the unknown model vector $\mathbf{x}$ to the vector of observations $\mathbf{d}$. In this case, 
we usually refer to g as the regularization term, where one or multiple terms are added to the objective function to 
promote certain features in the sought after solution and/or constraint it within a given set of allowed solutions.

Independently from the algorithm used to optimize such a cost function, a common feature of all proximal algorithms is
represented by the fact that one must be able to repeatedly evaluate the proximal operator of f and/or g. The proximal 
operator of a function f is defined as
…
Whilst evaluating a proximal operator does itself require solving an optimization problem, these subproblems often 
admit closed form solutions or can be solved very quickly with ad-hoc specialized methods. Several of such proximal 
operators are efficiently implemented in the PyProximal library.

Finally, there exists three main families of proximal algorithms that can be used to solve various flavors of equation 
\autoref{eq:problem} , namely:

- Proximal gradient method: this method, also commonly referred to as the FBS algorithm red, is usually the 
  preferred choice when A=I (I.e. identity). Accelerated versions such as the FISTA and TWist algorithms exist 
  and are usually preferred to the vanilla FBS method;
- Alternating direction of multipliers: this method is based on the well-known splitting strategy and can be used 
  for a broader class of problem than FBS and its accelarated versions. …
- Primal-dual: another popular algorithm able to tackle the generic problem in equation 1 with any choice of A. 
  It reformulates the original problem into its primal-dual version of solves a saddle optimization problem.

# Code structure

# Pedagogic example

Let's begin with a simple example showing how to compute the proximal operator of the L1 norm of a vector:

import numpy as np
from pyproximal import L1

l1 = L1()
x = np.arange(-5, 5, 0.1)
xp = l1.prox(x, 1)

As shown in line 4, one first initialized the operator. Subsequently, the prox method can be invoked to compute the proximal on a vector x with a given tau (here tau=1.). Note that this is a common pattern for all proximal operators in PyProximal

Moving on, we can how see how such a proximal operator can be used to solve a basic denoising problem of the form:


# Research applications

PyProximal has already been featured in a number of scientific publications:

- Joint inversion and segmentation of subsurface models: when inverting geophysical data for subsurface priorities, 
  a prior information that we would like to include in the inversion process is represented by the presence of a 
  discrete number of rock units, which can be parametrized in terms of their expected mean (or most likely value).  
  Ravasi and Birnie (2022) framed such a problem as a joint inversion and segmentation where the underlying optimization 
  is solved in alternating fashion using the Primal-dual algorithm.
- Plug-and-Play (PnP) priors: introduced in 2013 by Vent, in the PnP framework any proximal operator is interpreted 
  as a denoising problem and solved by means of any statistical or deep learning based denoiser. Recently, Romero

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References