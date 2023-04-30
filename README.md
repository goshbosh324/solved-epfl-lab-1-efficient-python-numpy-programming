Download Link: https://assignmentchef.com/product/solved-epfl-lab-1-efficient-python-numpy-programming
<br>
Efficient Python/NumPy Programming)

<h1>Introduction</h1>

For computational efficiency of typical operations in machine learning applications, it is very beneficial to use NumPy arrays together with vectorized commands, instead of explicit for loops. The vectorized commands are better optimized, and bring the performance of Python code (and similarly e.g. for Matlab) closer to lower level languages like C. In this exercise, you are asked to write efficient implementations for three small problems that are typical for the field of machine learning.

<h1>Getting Started</h1>

Follow the Python setup tutorial provided on our github repository here:

<a href="https://github.com/epfml/ML_course/tree/master/labs/ex01/python_setup_tutorial.md">github.com/epfml/ML</a> <a href="https://github.com/epfml/ML_course/tree/master/labs/ex01/python_setup_tutorial.md">course/tree/master/labs/ex01/python</a> <a href="https://github.com/epfml/ML_course/tree/master/labs/ex01/python_setup_tutorial.md">setup</a> <a href="https://github.com/epfml/ML_course/tree/master/labs/ex01/python_setup_tutorial.md">tutorial.md</a>

After you are set up, clone (using command line or a git desktop client) or download <a href="https://github.com/epfml/ML_course">the repository,</a> and start by filling in the template notebooks in the folder /labs/ex01, for each of the 3 tasks below.

To get more familiar with vector and matrix operations using NumPy arrays, it is also recommended to go through the npprimer.ipynb notebook in the same folder.

Note: The following three exercises could be solved by for-loops. While that’s ok to get started, the goal of this exercise sheet is to use the more efficient vectorized commands instead:

<h1>Useful Commands</h1>

We give a short overview over some commands that prove useful for writing vectorized code. You can read the full documentation and examples by issuing help(func).

At the beginning: import numpy as np

<ul>

 <li>a * b, a / b: element-wise multiplication and division of matrices (arrays) <em>a </em>and <em>b</em></li>

 <li>dot(b): matrix-multiplication of two matrices <em>a </em>and <em>b</em></li>

 <li>max(0): find the maximum element for each column of matrix <em>a </em>(note that NumPy uses zero-based indices, while Matlab uses one-based)</li>

 <li>max(1): find the maximum element for each row of matrix <em>a</em></li>

 <li>mean(a), np.std(a): compute the mean and standard deviation of all entries of <em>a</em></li>

 <li>shape: return the array dimensions of <em>a</em></li>

 <li>shape[k]: return the size of array <em>a </em>along dimension <em>k</em></li>

 <li>sum(a, axis=k): sum the elements of matrix <em>a </em>along dimension <em>k</em></li>

 <li>inv(a): returns the inverse of a square matrix <em>a</em></li>

</ul>

A broader tutorial can be found here: <a href="http://www.engr.ucsb.edu/~shell/che210d/numpy.pdf">http://www.engr.ucsb.edu/</a><a href="http://www.engr.ucsb.edu/~shell/che210d/numpy.pdf">~</a><a href="http://www.engr.ucsb.edu/~shell/che210d/numpy.pdf">shell/che210d/numpy.pdf</a>

For users who were more familiar with Matlab, a nice comparison of the analogous functions can be found here:

<a href="https://numpy.org/devdocs/user/numpy-for-matlab-users.html">https://numpy.org/devdocs/user/numpy-for-matlab-users.html</a>

Figure 1: Two sets of points in the plane. The circles are a subset of the dots and have been perturbed randomly.

<h1>Task A: Matrix Standardization</h1>

The different dimensions or features of a data sample often show different variances. For some subsequent operations, it is a beneficial preprocessing step to standardize the data, i.e. subtract the mean and divide by the standard deviation for each dimension. After this processing, each dimension has zero mean and unit variance. Note that this is not equivalent to data whitening, which additionally de-correlates the dimensions (by means of a coordinate rotation).

Write a function that accepts data matrix <em>x </em>∈ R<em><sup>n</sup></em><sup>×<em>d </em></sup>as input and outputs the same data after normalization. <em>n </em>is the number of samples, and <em>d </em>the number of dimensions, i.e. rows contain samples and columns features.

<h1>Task B: Pairwise Distances in the Plane</h1>

One application of machine learning to computer vision is interest point tracking. The location of corners in an image is tracked along subsequent frames of a video signal (see Figure 1 for a synthetic example). In this context, one is often interested in the pairwise distance of all points in the first frame to all points in the second frame. Matching points according to minimal distance is a simple heuristic that works well if many interest points are found in both frames and perturbations are small.

Write a function that accepts two matrices <strong>P </strong>∈ R<em><sup>p</sup></em><sup>×2</sup><em>,</em><strong>Q </strong>∈ R<em><sup>q</sup></em><sup>×2 </sup>as input, where each row contains the (<em>x,y</em>) coordinates of an interest point. Note that the number of points (<em>p </em>and <em>q</em>) do not have to be equal. As output, compute the pairwise distances of all points in <strong>P </strong>to all points in <strong>Q </strong>and collect them in matrix <strong>D</strong>. Element <em>D<sub>i,j </sub></em>is the Euclidean distance of the <em>i</em>-th point in <strong>P </strong>to the <em>j</em>-th point in <strong>Q</strong>.

<h1>Task C: Likelihood of a Data Sample</h1>

In this exercise, you are not required to understand the statistics and machine learning concepts described here yet. The goal here is just to practically implement the assignment of data to two given distributions, in Python.

A subtask of many machine learning algorithms is to compute the likelihood <em>p</em>(<em>x</em><em><sub>n</sub></em>|<em>θ</em>) of a sample <em>x</em><em><sub>n </sub></em>for a given density model with parameters <em>θ</em>. Given <em>k </em>models, we now want to assign <em>x</em><em><sub>n </sub></em>to the model for which the likelihood is maximal: <em>a<sub>n </sub></em>= argmax<em><sub>m </sub>p</em>(<em>x</em><em><sub>n </sub></em>|<em>θ</em><em><sub>m</sub></em>), where <em>m </em>= 1<em>,…,k</em>. Here <em>θ</em><em><sub>m </sub></em>= (<em>µ</em><em><sub>m</sub></em><em>,</em><strong>Σ</strong><em><sub>m</sub></em>) are the parameters of the <em>m</em>-th density model (<em>µ</em><em><sub>m </sub></em>∈ R<em><sup>d </sup></em>is the mean, and <strong>Σ</strong><em><sub>m </sub></em>is the so called covariance matrix).

We ask you to implement the assignment step for the two model case, i.e. <em>k </em>= 2. As input, your function receives a set of data examples <em>x</em><em><sub>n </sub></em>∈ R<em><sup>d </sup></em>(indexed by 1 ≤ <em>n </em>≤ <em>N</em>) as well as the two sets of parameters <em>θ</em><sub>1 </sub>= (<em>µ</em><sub>1</sub><em>,</em><strong>Σ</strong><sub>1</sub>) and <em>θ</em><sub>2 </sub>= (<em>µ</em><sub>2</sub><em>,</em><strong>Σ</strong><sub>2</sub>) of two given multivariate Gaussian distributions:

<em>.</em>

|<strong>Σ</strong>| is the determinant of <strong>Σ </strong>and <strong>Σ</strong><sup>−1 </sup>its inverse. Your function must return the ’most likely’ assignment <em>a<sub>n </sub></em>∈ {1<em>,</em>2} for each input point <em>n</em>, where <em>a<sub>n </sub></em>= 1 means that <em>x</em><em><sub>n </sub></em>has been assigned to model 1. In other words in the case that <em>a<sub>n </sub></em>= 1, it holds that <em>p</em>(<em>x</em><em><sub>n </sub></em>|<em>µ</em><sub>1</sub><em>,</em><strong>Σ</strong><sub>1</sub>) <em>&gt; p</em>(<em>x</em><em><sub>n </sub></em>|<em>µ</em><sub>2</sub><em>,</em><strong>Σ</strong><sub>2</sub>).

2

<h1>Theory Questions</h1>

In addition to the practical exercises you do in the labs, as for example above, we will in future labs also provide you some theory oriented questions, to prepare you for the final exam. As the rest of the exercises, it is <em>not </em>mandatory to solve them, but we would recommend that you at least look at – and try – some of them during the semester. From last year’s experience, many students where surprised by the heavy theoretical focus of the final exam after having worked on the two very practical projects. Do not fall for this trap! Passing the course require acquiring both a practical and a theoretical understanding of the material, and those exercises should help you with the latter.

However, please note that the difficulty of these exercises might not be of the same level as the exam and are not enough by themselves; you should read the additional material given at the end of the lectures and do the exercises in the recommended books – see <a href="https://github.com/epfml/ML_course/raw/master/lectures/course_info_sheet.pdf">The course info sheet</a><a href="https://github.com/epfml/ML_course/raw/master/lectures/course_info_sheet.pdf">.</a>

Note that we will try to, but might not, provide solutions.

This week, as we just started the course, there are no exercises. You should refresh your mind of the prerequisites, especially on the following topics.

<ul>

 <li>Make sure your linear algebra is fresh in memory, especially

  <ul>

   <li>Matrix manipulation (<a href="https://en.wikipedia.org/wiki/Matrix_multiplication">Multiplication</a><a href="https://en.wikipedia.org/wiki/Matrix_multiplication">,</a> <a href="https://en.wikipedia.org/wiki/Transpose">Transpose</a><a href="https://en.wikipedia.org/wiki/Transpose">,</a> <a href="https://en.wikipedia.org/wiki/Invertible_matrix">Inverse</a><a href="https://en.wikipedia.org/wiki/Invertible_matrix">)</a></li>

   <li><a href="https://en.wikipedia.org/wiki/Rank_(linear_algebra)">Ranks</a><a href="https://en.wikipedia.org/wiki/Rank_(linear_algebra)">,</a> <a href="https://en.wikipedia.org/wiki/Linear_independence">Linear independence</a></li>

   <li><a href="https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors">Eigenvalues and Eigenvectors</a></li>

  </ul></li>

</ul>

You can use the following resources to help you get up to speed if needed.

<ul>

 <li>The <a href="https://github.com/epfml/ML_course/raw/master/lectures/handout_linalg_book.pdf">Linear Algebra handout</a></li>

 <li>Gilbert Strang’s <a href="http://math.mit.edu/~gs/learningfromdata/">Linear Algebra and Learning from Data</a> or <a href="http://math.mit.edu/~gs/linearalgebra/">Introduction to Linear Algebra</a><a href="http://math.mit.edu/~gs/linearalgebra/">.</a> Some chapters are available online, and the books (along with many other textbooks on linear algebra) should be available at the EPFL Library.</li>

</ul>

<ul>

 <li>If it has been long since your last calculus class, make sure you know how to handle <a href="https://en.wikipedia.org/wiki/Matrix_calculus">Gradients</a><a href="https://en.wikipedia.org/wiki/Matrix_calculus">.</a> You can find a quick summary and useful identities in <a href="http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf">The Matrix Cookbook</a><a href="http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf">.</a></li>

 <li>For probability and statistics, you should at least know about

  <ul>

   <li><a href="https://en.wikipedia.org/wiki/Conditional_probability_distribution">Conditional</a> and <a href="https://en.wikipedia.org/wiki/Joint_probability_distribution">joint</a> probability distributions</li>

   <li><a href="https://en.wikipedia.org/wiki/Bayes%27_theorem">Bayes theorem</a></li>

   <li><a href="https://en.wikipedia.org/wiki/Random_variable">Random variables</a><a href="https://en.wikipedia.org/wiki/Random_variable">,</a> <a href="https://en.wikipedia.org/wiki/Independence_(probability_theory)">independence</a><a href="https://en.wikipedia.org/wiki/Independence_(probability_theory)">,</a> <a href="https://en.wikipedia.org/wiki/Variance">variance</a><a href="https://en.wikipedia.org/wiki/Variance">,</a> <a href="https://en.wikipedia.org/wiki/Expected_value">expectation</a></li>

   <li>The <a href="https://en.wikipedia.org/wiki/Normal_distribution">Gaussian distribution</a></li>

  </ul></li>

</ul>

If you need a refresh, check Chapter 2 in Pattern Recognition and Machine Learning by Christopher Bishop, available at the EPFL Library.