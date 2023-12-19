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
It has already been featured in a number of scientific publications XX and in a graduate-level course on inverse problems XX. 
As the adoption of the library grows across many disciplies, we believe that `PyProximal` will enable exciting scientific discoveries 
in a variety of scientific problems with societal impact.

# Mathematical framework

Convex optimization is routinely used to solve problems of the form:

\begin{equation}
\label{eq:problem}
\min_\mathbf{x} f(\mathbf{x}) +g(\mathbf{Lx})
\end{equation}


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