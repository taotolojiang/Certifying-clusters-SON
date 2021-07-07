module ClusterTests
import ..Einsum.@einsumnum
using LinearAlgebra
using Random
using Distributions
using LightGraphs

sumdim2(w) = reshape(sum(w, dims=2), size(w,1))

function find_cluster(U, precision)
    p, n = size(U)
    l_set = collect(1:n)
    C = zeros(Int, n)
    c = 1

    while length(l_set) > 0
    	m = length(l_set)
    	l1 = l_set[1]
    	C[l1] = c
    	for j = 2:m
    		l2 = l_set[j]
    		if norm(U[:,l1] - U[:,l2]) <= precision
    			C[l2] = c
    		end
    	end
    	l_set = findall(C .== 0)
    	c += 1
    end
    return C
end

function find_cluster_admm(mx_matrix)
    n, n = size(mx_matrix)
    C = zeros(Int, n)
    g = SimpleGraph(mx_matrix)
    con_components = connected_components(g)
    c = 1
    for component in con_components
        C[component] .= c
        c += 1
    end
    return C
end


function cluster_test(X, G_mat, Y_mat, Omega, a, gamma, tau, C_cand_pt, weight)
    d, n = size(X)
    C_cand_pt = reshape(C_cand_pt, :, )
    C_cand = unique(C_cand_pt)
    clustering_cond = true    
    weighted_X = X * Diagonal(weight)

    # check the separation condition
    for C_ind1 = 1:length(C_cand)
        c1 = C_cand[C_ind1]
        C1_pt = findall(isequal(c1), C_cand_pt)
        m = length(C1_pt)
        if m != length(C_cand_pt)
            for C_ind2 = C_ind1 + 1 : length(C_cand)
                c2 = C_cand[C_ind2]
                C2_pt = findall(isequal(c2), C_cand_pt)
                C1C2 = vcat(C1_pt, C2_pt)
                sum_weighted_X = sumdim2(weighted_X[:, C1C2])
                sum_weight = sum(weight[C1C2])
                avg_weighted_X = sum_weighted_X ./ sum_weight
                tmp = (X[:, C1C2] .- repeat(avg_weighted_X, 1, length(C1C2))).^2
                squared_dist = sum((X[:, C1C2] .- repeat(avg_weighted_X, 1, length(C1C2))).^2 * Diagonal(weight[C1C2]))
                if squared_dist <= tau
                    C_cand_pt[C1C2] .= 0
                    clustering_cond = false
                end
            end
        end
    end

    
    for C_ind = 1:length(C_cand)

        c = C_cand[C_ind]
        notC_ind = setdiff(C_cand, C_ind)
        sameC = findall(isequal(c), C_cand_pt)
        weight_sum = sum(weight[sameC])
        diffC = setdiff(1:n, sameC)
        m = length(sameC)

                
        Chiquet_mult_mat = zeros(m, m, d);
        Chiquet_mult = zeros(d,)
        x_omega_diff = zeros(d)

        l1 = 1
        l2 = 1

        # check the CGR subgradients
        if m > 1
            gij = zeros(d)
            gik_sum = zeros(d)
            gjk_sum = zeros(d)


            for pos1 = 1:m, pos2 = pos1+1:m
                l1 = sameC[pos1]
                l2 = sameC[pos2]

                gij[:] = reshape(G_mat[l1, l2, :], :, )
                x_omega_diff[:] .= 1.0/weight_sum .* (X[:, l1] .- X[:, l2] .- Omega[:, l1] .+ Omega[:, l2])
                gik_sum[:] = reshape(sum(Diagonal(weight[diffC]) * G_mat[l1, diffC, :], dims = 1), :, )
                gjk_sum[:] = reshape(sum(Diagonal(weight[diffC]) * G_mat[l2, diffC, :], dims = 1), :, )
                Chiquet_mult[:] .= gij .+ x_omega_diff .+ 1.0/weight_sum .* (gik_sum .- gjk_sum)

                if norm(Chiquet_mult) <= gamma
                    Chiquet_mult_mat[pos1, pos2, :] .= Chiquet_mult
                    Chiquet_mult_mat[pos2, pos1, :] .= - Chiquet_mult
                else
                    clustering_cond = false
                    #println("CGR fails")
                    C_cand_pt[sameC] .= 0
                end

            end
        end
    end
    return clustering_cond, C_cand_pt
end


function admm_Chi_Lange_wtest(X, gamma, weight, maxit)
	p, n = size(X)
	numterms = div(n * (n - 1), 2)
	index_set = zeros(Int, 2, numterms)
	pos = 1
	
	nu = 1.0
    weight_l = zeros(numterms)


    for i = 1 : n - 1
        for j = i + 1 : n
            index_set[1,pos] = i
            index_set[2,pos] = j
            weight_ij = weight[i] * weight[j]
            weight_l[pos] = weight_ij
            pos += 1
        end
    end

	scfac = norm(X)

    sigma = gamma / nu
    X_sum = sumdim2(X * Diagonal(weight))
    X_bar = repeat(X_sum, 1, n)

    weight_sum = sum(weight)

    U = zeros(p, n)
    V = zeros(p, numterms)
    y_term = zeros(p, numterms)
    Lambda = zeros(p, numterms)
    Lambda_sum = zeros(p)
    Unew = zeros(p, n)
    Vnew = zeros(p, numterms)
    U_diff = zeros(p)
    s_term = zeros(p, numterms)
    Lambdanew = zeros(p, numterms)
    Optimality_cond = zeros(p, n)

    z = zeros(p,)    
    s = zeros(p, n)
    r = zeros(p, numterms)
    Y = zeros(p, n)

    G_mat = zeros(n, n, p)
    Y_mat = zeros(n, n, p)
    
    X_s = zeros(p, n)
    Y_s = zeros(p, numterms)
    Z_s = zeros(p, n)
    s_s = zeros(1, n)
    u_s = zeros(1, n)
    t_s = zeros(1, numterms)
    Alpha_s = zeros(p, numterms)
    Beta_s = zeros(p, n)
    gamma_s = zeros(1, n)
    Optimality_cond = zeros(p, n)
    mu = 0.0
    clusterID = zeros(Int, n)

    l1_set = Vector{Vector{Int}}()
	l2_set = Vector{Vector{Int}}()
	for i = 1:n
    	push!(l1_set, findall(index_set[1,:] .== i))
    	push!(l2_set, findall(index_set[2,:] .== i))
	end

	tcount = 1
    clustering_cond = false;
    C_cand_pt = zeros(Int, n)
    loopsize = 8
    itcount = 0

    mx_array = Vector{Int}()
    mx_matrix = zeros(n, n)

    while clustering_cond==false
    
        for count = 1:loopsize

            # update U, V, Lambda
            y_term .= (Lambda .+ nu.*V) * Diagonal(weight_l)
            for i = 1 : n
                wi = weight[i]
                Y[:, i] .= X[:, i] .+
                    (1/wi) * sumdim2(y_term[:,l1_set[i]]) .-
                    (1/wi) * sumdim2(y_term[:,l2_set[i]])
            end
            Unew .= (1/(1+weight_sum*nu)).*Y .+ (nu/(1+weight_sum*nu)).*X_bar

            nurecip = 1/nu

            mx_array = Vector{Int}()

            for l = 1 : numterms
                @einsumnum Y_s[~1,l] = Unew[~1,index_set[1,l]] - Unew[~1,index_set[2,l]]
                @einsumnum z[~1] = Y_s[~1,l] - nurecip*Lambda[~1,l]
                mx = max(0.0, 1-sigma/(norm(z)))
                if mx == 0
                    push!(mx_array, l)
                end
                @einsumnum Vnew[~1,l] =  mx * z[~1]
                @einsumnum r[~1,l] = Y_s[~1,l] - Vnew[~1,l]
                @einsumnum Lambdanew[~1,l] = Lambda[~1,l] - nu * r[~1,l]
            end
            
            U .= Unew
            V .= Vnew
            Lambda .= Lambdanew
        end

        X_s = U;
        Z_s = X_s .- X
        s_s = 0.5 * (ones(1, n) .+ sum(Z_s.^2, dims = 1))
        u_s = 0.5 * (- ones(1, n) .+ sum(Z_s.^2, dims = 1))
        t_s = sqrt.(sum(Y_s.^2, dims = 1))
        Alpha_s = Lambda
		
		Beta_s = zeros(p, n)
        for i = 1:n
            wi = weight[i]
            Beta_s[:,i] = - (1/wi) * sumdim2(Alpha_s[:,l1_set[i]] * Diagonal(weight_l[l1_set[i]])) .+ 
                (1/wi) * sumdim2(Alpha_s[:,l2_set[i]] * Diagonal(weight_l[l2_set[i]]))
        end
        gamma_s = 0.5 .* (ones(1, n) - sum(Beta_s.^2, dims = 1))

        fx = sum(weight' .* s_s) + gamma * sum(weight_l' .* t_s)
        halpha = tr(transpose(X * Diagonal(weight)) * Beta_s) + sum(gamma_s' .* weight)
        mu = fx - halpha
        mu = max(mu, 1e-15 * scfac^2)
        #println("mu = ", mu)
        for l = 1 : size(index_set, 2)
            G_mat[index_set[1,l], index_set[2,l], :] .= -Alpha_s[:, l]
            G_mat[index_set[2,l], index_set[1,l], :] .= Alpha_s[:, l]
            Y_mat[index_set[1,l], index_set[2,l], :] .= Y_s[:, l]
            Y_mat[index_set[2,l], index_set[1,l], :] .= -Y_s[:, l]
        end

        Sigma2 = repeat(s_s, p, 1) .* Beta_s .+ repeat(ones(1, n) .- gamma_s, p, 1) .* Z_s
        sigma3 = s_s .* gamma_s .+ (ones(1, n) .- gamma_s) .* u_s
        Omega = (sigma3 ./ s_s) .* Z_s .+ (ones(1, n) ./ s_s) .* Sigma2

        tau = 2 * mu

        num_pair = length(mx_array)
        mx_matrix = zeros(n, n)
        for l = 1:num_pair
            mx_pair_ind = mx_array[l]
            mx_matrix[index_set[1, mx_pair_ind], index_set[2, mx_pair_ind]] = 1
            mx_matrix[index_set[2, mx_pair_ind], index_set[1, mx_pair_ind]] = 1
        end
        #C_cand_pt = find_cluster(X_s, mu^0.5)
        C_cand_pt = find_cluster_admm(mx_matrix)
        clustering_cond, C_cand_pt = cluster_test(X_s, G_mat, Y_mat, Omega, X, gamma, tau, C_cand_pt, weight)

        tcount = tcount + 1
        itcount += loopsize
        if itcount >= maxit
            break
        end
    end

    min_strict_comp = Inf
    for l = 1 : size(index_set,2)
        strict_comp = (t_s[l] + gamma)  -
            sqrt(sum((Alpha_s[:,l] + Y_s[:,l]).^2))
        min_strict_comp = min(strict_comp, min_strict_comp)
    end

    clusterID = C_cand_pt
    #println(mx_matrix)
    return X_s, Y_s, Z_s, s_s, u_s, t_s, Alpha_s, Beta_s, gamma_s, Optimality_cond, mu, clusterID, itcount, clustering_cond, min_strict_comp
end




end