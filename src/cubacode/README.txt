# Notice
This code was modified in 2018 to be compatible with the source code for the paper [Latent-space Dynamics for Reduced Deformable Simulation](https://www.lawsonfulton.com/publication/latent-space-dynamics/)

It still falls under the original license found in COPYING.txt

# Original
Greetings,

This source code release provides an abstracted implementation of the cubature optimization algorithm described in [An et al. 2008]. The class is GreedyCubop, and to use it, implement the two pure virtual functions for your own problem. "evalPointForceDensity" should evaluate the reduced force density at a given sample point, which are the 'g' vectors in equation (12) of the paper. You should probably override "handleCubature" as well, to write the cubature rules to disk for run-time use. Then, call "run" with your own parameters to run the algorithm. "runNQP" uses an alternate method to solve the NNLS problem using quadratic programming, so feel free to try that - it may be faster for certain problems. Code to use the resulting cubature rules at run-time is not provided, but it's relatively trivial (equation (5) in the paper).

Please let me know if you use this or encounter problems. I'd love to hear from anyone that takes interest in this work.

Thank you,

Steven An (stevenan@cs.cornell.edu)
March 15th, 2010

References:
Steven S. An, Theodore Kim and Doug L. James, Optimizing Cubature for Efficient Integration of Subspace Deformations, ACM Transactions on Graphics (SIGGRAPH ASIA Conference Proceedings), 27(5), December 2008, pp. 164:1-164:11.
