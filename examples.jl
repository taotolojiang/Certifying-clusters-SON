include("einsum3.jl")
include("./data_generation.jl")
include("./cluster_test.jl")
# include("../../GitHub/Einsum.jl/src/einsum2.jl")
import .ClusterTests
import .DataGeneration
# n = 500
# ang = pi/5
# noise = 0.05
# gaussian = false
# lambdamin = .1
# lambdamax = .5
# numlambda = 250
# lambdavec = collect(range(lambdamin, stop=lambdamax, length=numlambda))
# outfilename = "halfmoon_pi5_test5t.txt"
# maxit = 50000
# t0 = time()
# DataGeneration.run_test_lambda_halfmoon(n, ang, noise, gaussian,
#                                       lambdavec, 10,
#                                       outfilename, maxit)
# print(time() - t0)


"""
Apply sum-of-norms clustering to a mixture of Gaussians
"""
n = 500    # number of points
d = 2     # dimension
k = 2    # number of Gaussians
# sigma = 1.0    # standard deviation is sigma * sqrt(2)
theta = 0.7    # we only count Rand index for data points that are located at 0.5 * theta * min mean distance away from the respective means
lambdamulseq = 0.1:0.05:1.2    # the multiple of the lambda value which is supposed to recover Gaussians exactly from the theorem
meandistvec = 1:6    # the range of mean distance, mean distance is the ratio between the pairwise distance of Gaussian means and standard deviation
maxit = 50000    # maximum iteration
outfilename = "test_lambda3.txt"
t0 = time()
DataGeneration.run_test_lambda_rho_meandistance(n, d, k, theta, lambdamulseq, meandistvec, maxit, outfilename)
print(time() - t0)
