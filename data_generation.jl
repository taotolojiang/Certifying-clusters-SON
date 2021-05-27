module DataGeneration
include("./einsum3.jl")
import ..Einsum.@einsumnum
include("./cluster_test_simods.jl")
import .ClusterTests
using LinearAlgebra
using Random
using Distributions
using LightGraphs

"""
Inputs:
mu: d-by-k, each column j is the coordinate of centroid for Gaussian j;
sigma: 1-by-k, each entry j is the standard deviation of Gaussian j (we
assume the covariance matrix of component j is sigma_j * identity matrix;
w: 1-by-k, mixture probabilities;
n: number of samples
Outputs:
a: d-by-n, data of SON;
C_theta: n-by-1, the value of entry j is the assignment of j to its
Gaussian component i, the value is 0 if point j is > 0.5 * theta * min pairwise distance
among the means, away from its mean;
Vm: the list of indices of points which are within theta standard
deviation from its component mean
"""

function  gm_generator(mu, sigma, w, n, theta)

    rng = MersenneTwister(345)
    d, k = size(mu)
    compID = zeros(Int, n)
    mu_vec = zeros(d, n)
    sigma_vec = zeros(n)
    weight = zeros(n)

    x = rand(rng, n)
    for j = 1 : k
        if j == 1
            compID[x .<= w[1]] .= 1
        else
            compID[(x .> sum(w[1 : j - 1])) .&  (x .<= sum(w[1 : j]))] .= j
        end
        mu_vec[:, compID .== j] .= repeat(mu[:,j], 1, sum(compID .== j))
        sigma_vec[compID .== j] .= sigma[j]
    end
    y = randn(rng, (d, n))
    a = y .* repeat(sigma_vec', d, 1) .+ mu_vec
    Id = Matrix{Float64}(I,2,2)
    distr = MvNormal([0.0,0.0], Id)
    
    # a = repeat(a, 1, 2)
    # mu_vec = repeat(mu_vec, 1, 2)
    # sigma_vec = repeat(sigma_vec, 1, 2)
    # compID = repeat(compID, 2, 1)
    # weight = 50 .* pdf(distr, y)
    # weight = repeat(weight, 2, 1)
    weight = 100 .* pdf(distr, y)
    weight = weight[:]
    n = n

    C_theta = copy(compID)
    for j = 1: n
        if norm(a[:,j] - mu_vec[:, j]) > theta * sigma_vec[j]
            C_theta[j] = 0
        end
    end
    Vm = findall(C_theta .!= 0 )

    return a, compID, Vm, C_theta, weight
end

function twoGaussianspirals(N, std, noise)
    rng = MersenneTwister(345)
    N1 = floor(N/2)
    N2 = N-N1
    N1 = convert(Int64, N1)
    N2 = convert(Int64, N2)
    
    u1 = randn(rng, N1, 1)
    u2 = randn(rng, N2, 1)
    gamma = std
    
    u1 = randn(rng, N1, 1)
    d1 = [cos.(gamma .* u1) .+ rand(rng, N1, 1) .* noise sin.(gamma .* u1) .+ rand(rng, N1, 1) .* noise]

    u2 = randn(rng, N2, 1)
    d2 = [-cos.(gamma .* u2) .+ rand(rng, N2, 1) .* noise sin.(gamma .* u2) .+ rand(rng, N2, 1) .* noise .+ 1]
    
    a = transpose(vcat(d1, d2))
    compID = vcat(ones(Int, N1, 1), 2 .* ones(Int, N2, 1))
    
    return a, compID
end

function twohalfmoons(N, noise, std, amt)
    rng = MersenneTwister(345)
    N1 = floor(N/2)
    N2 = N-N1
    N1 = convert(Int64, N1)
    N2 = convert(Int64, N2)
    
    u1 = rand(rng, N1, 1)
    u2 = rand(rng, N2, 1)
    
    gamma = pi
    u1 = rand(rng, N1, 1) .- 0.5
    d1 = [cos.(gamma .* u1) .+ (rand(rng, N1, 1) .- 0.5) .* noise sin.(gamma .* u1) .+ (rand(rng, N1, 1) .- 0.5) .* noise]

    u2 = rand(rng, N2, 1) .- 0.5
    d2 = [-cos.(gamma .* u2) .+ (rand(rng, N2, 1) .- 0.5) .* noise sin.(gamma .* u2) .+ (rand(rng, N2, 1) .- 0.5) .* noise .+ 1]
    
    distr = Normal(0, std)
    f1 = amt .* pdf.(distr, gamma .* u1)
    f2 = amt .* pdf.(distr, gamma .* u2)

    a = transpose(vcat(d1, d2))
    weight = vcat(f1, f2)
    weight = weight[:]
    compID = vcat(ones(Int, N1, 1), 2 .* ones(Int, N2, 1))
    return a, compID, weight
end


"""
n = 80    # number of points
d = 2     # dimension
k = 2    # number of Gaussians
# sigma = 1.0    # standard deviation is sigma * sqrt(2)
theta = 0.7    # we only count Rand index for data points that are located at 0.5 * theta * min mean distance away from the respective means
lambdamulseq = 0.1:0.1:0.8    # the multiple of the lambda value which is supposed to recover Gaussians exactly from the theorem
meandistvec = 1:6    # the range of mean distance, mean distance is the ratio between the pairwise distance of Gaussian means and standard deviation
maxit = 10000    # maximum iteration
outfilename = "test_lambda3.txt"

"""
function run_test_lambda_rho_meandistance(n, d, k, theta, lambdamulseq, meandistvec, maxit, outfilename)
    open(outfilename, "w") do oh
        println(oh, "n = ", n, " d = ", d, " k = ", k, " theta = ", theta, "\n")
        sigma = sqrt(2)
        for meandist in meandistvec
            ee = Matrix{Float64}(meandist * I,d,d)    
            mu = ee[1:k, :]
            println(meandist)
            a, compID, Vm, C_theta, weight = gm_generator(mu,
                                                  sigma * ones(k),
                                                  ones(k) / k,
                                                  n,
                                                  theta)

            best_Randind_vec = Vector{Float64}()
            best_Randind_vec_s = Vector{Float64}()
            best_lambda_vec = Vector{Float64}()
            best_lambda_vec_s = Vector{Float64}()
            average_weight = sum(weight) / n
            theta_ineq = 0.7 * 0.5 * meandist
            chicum = cdf(Chi(d), theta_ineq)
            lambdamin = 2 * theta_ineq * sigma * k / (chicum * n * average_weight * 100) # the extra term 5 in the denominator is to ensure lambdamin < lambdamax
            println(lambdamin)
            lambdamax = meandist * sqrt(2) / (2 * (n - 1) * average_weight * 0.1) # the extra term 0.2 in the denominator is to ensure lambdamin < lambdamax
            println(lambdamax)
            println(oh, "------")
            println(oh, "lambdamin = ", lambdamin, " lambda_max = ", lambdamax)
            @assert lambdamin<lambdamax
            lambdamid = (lambdamin + lambdamax) / 2
            lambdavec = lambdamulseq * lambdamid
            Randind_vec = Vector{Float64}()
            Randind_vec_s = Vector{Float64}()
            num_point_s = 0
            for lambda in lambdavec
                if lambda <= 0
                    continue
                end
                X_s, Y_s, Z_s, s_s, u_s, t_s, Alpha_s, Beta_s, gamma_s, Optimality_cond, gap, clusterID, itcount, 
                    clustering_cond, min_strict_comp = ClusterTests.admm_Chi_Lange_wtest(a, lambda, weight, maxit)
                num_clust, num_distinct_clust, num_inconclusive = ClusterTests.count_correctly_clustered_general(k, clusterID, compID) 
                Randind, inda, indb, indc, indd = ClusterTests.compute_randind(compID, clusterID)
                Randind_s, inda_s, indb_s, indc_s, indd_s = ClusterTests.compute_randind(compID[Vm], clusterID[Vm])
                push!(Randind_vec, Randind)
                push!(Randind_vec_s, Randind_s)
                num_point_s = length(Vm)
                println(oh, "lambda = ", lambda, " duality gap = ", gap, 
                    " \n  num_clust = ", num_clust, " numvm_total = ",
                    n, " num_distinct_clust = ", num_distinct_clust, " num_inconclusive = ", num_inconclusive,
                    "\n clusters = ", clusterID)
                println(oh, "  itcount = ", itcount, " maxit = ", maxit)
                println(oh, "  clustering condition is ", clustering_cond)
                println(oh, "Rand index = ", Randind, ";")
                println(oh, "The smaller dataset has ", num_point_s, " points;")
                    println(oh, "Rand index for a smaller data set = ", Randind_s, ";")
                if length(unique(clusterID)) == 1 && num_inconclusive == 0
                    println(oh, "We have reached the max lambda!")
                    break
                end
            end
            bestrand_val, bestrand_location = findmax(Randind_vec)
            push!(best_Randind_vec, bestrand_val)
            push!(best_lambda_vec, lambdavec[bestrand_location])
            bestrand_val_s, bestrand_location_s = findmax(Randind_vec_s)
            push!(best_Randind_vec_s, bestrand_val_s)
            push!(best_lambda_vec_s, lambdavec[bestrand_location_s])
            println(oh, "The means are located ", meandist, " standard deviation away from each other. \n")
            println(oh, "The lambda values with best Rand index are ", best_lambda_vec,
                "\n The corresponding Rand indices are ", best_Randind_vec,
                "\n The lambda values with best Rand index of a smaller data set are ", best_lambda_vec_s,
                "\n The corresponding Rand indices of smaller data set are ", best_Randind_vec_s)
            println(oh, "The lambda values are ", lambdavec,
                "\n The Rand indices are ", Randind_vec,
                "\n The Rand idices for a smaller data set are ", Randind_vec_s)
            println(oh, "=======================================")
        end
    end
    nothing
end



"""
half moon dataset
"""

function run_test_lambda_halfmoon(n, std, noise, Gaussian, lambdamulseq, amtvec, outfilename, maxit)

    open(outfilename, "w") do oh

        k = 2
        numlambda = length(lambdamulseq)
        for amt in amtvec
            if Gaussian == true
                a, compID = twoGaussianspirals(n, std, noise)
            else
                a, compID, weight = twohalfmoons(n, noise, std, amt)
            end

            lambdamid = 0.05/amt
            lambdavec = lambdamulseq * lambdamid

            println(oh, "the data is:")
            println(oh, a)
            println(oh, "the weights are as follows:")
            println(oh, weight)
            println(oh, "------")
            println(oh, "n = ", n)
            # println(oh, "lambdamin = ", lambdamin, " lambda_max = ", lambdamax)
            println(oh, "the lambda vector is ", lambdavec)
            #weight = ones(n)
            #weight=weight_generator(a, rho, amt)
            mu_vec = zeros(numlambda)
            Randind_vec = zeros(numlambda)
            weighted_Randind_vec = zeros(numlambda)
            num_clust_vec = zeros(Int, numlambda,)
            num_distinct_clust_vec = zeros(Int, numlambda,)
            num_distinct_clust_vec = zeros(Int, numlambda,)
            itcount_all = zeros(Int, numlambda)
            clusteringcond_all = zeros(Bool, numlambda)
            min_strict_comp_all = zeros(numlambda)
            println(oh, "===============")
            println(oh, "The boosted amount is ", amt)
            for j = 1:numlambda
                lambda = lambdavec[j]
                X_s, Y_s, Z_s, s_s, u_s, t_s, Alpha_s, Beta_s, gamma_s, Optimality_cond, gap, clusterID, itcount, 
                            clustering_cond, min_strict_comp = ClusterTests.admm_Chi_Lange_wtest(a, lambda, weight, maxit)
                num_clust, num_distinct_clust, num_inconclusive = ClusterTests.count_correctly_clustered_general(k, clusterID, compID) 
                Randind, inda, indb, indc, indd = ClusterTests.compute_randind(compID, clusterID)
                weighted_Randind, weighted_inda, weighted_indb, weighted_indc, weighted_indd = ClusterTests.compute_weighted_randind(compID, clusterID, weight)


                println(oh, "lambda = ", lambda, " duality gap = ", gap)
                println(oh, "Rand index = ", Randind, "weighted Rand index = ", weighted_Randind, ";")
                println(oh, "Same half moon, same cluster = ", inda, "; different half moons, different clusters = ", indb, ";")
                println(oh, "same half moon, different clusters = ", indc, "; different half moons, same cluster = ", indd)
                print(oh, " num_clust = ", num_clust, " numvm_total = ",
                        num_clust, " num_distinct_clust = ", num_distinct_clust,
                        "\n clustering = ", clusterID,
                        "\n clustering condition = ", clustering_cond)
                println(oh, "  itcount = ", itcount, " maxit = ", maxit)
                println(oh, "-------------------------------------")
            
                #println(oh, "X=", X_s)
                
                mu_vec[j] = gap
                Randind_vec[j] = Randind
                weighted_Randind_vec[j] = weighted_Randind
                num_clust_vec[j] = num_clust
                num_distinct_clust_vec[j] = num_distinct_clust   
                itcount_all[j] = itcount
                clusteringcond_all[j] = clustering_cond
                min_strict_comp_all[j] = min_strict_comp

                if length(unique(clusterID)) == 1 && num_inconclusive == 0
                    println(oh, "We have reached the max lambda!")
                    break
                end
            end
            println(oh, "========================================")
            println(oh, "lambda = ", lambdavec, ";")
            println(oh, "Rand index = ", Randind_vec, ";")
            println(oh, "weighted Rand index = ", weighted_Randind_vec, ";")
            println(oh, "duality_gap_mu = ", mu_vec, ";")
            println(oh, "number_of_clustered_points = ", num_clust_vec, ";")
            println(oh, "number_of_distinct_clusters = ", num_distinct_clust_vec, ";")
            println(oh, "itcount_all = ", itcount_all, ";")
            println(oh, "clusteringcond_all = ",
                    [Int(clusteringcond_all[j]) for j = 1 : numlambda], ";")
            println(oh, "min_strict_comp_all = ", min_strict_comp_all, ";")
        end

    end
    nothing    
end


end







