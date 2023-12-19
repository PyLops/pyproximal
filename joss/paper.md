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


# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

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