"""
Compute the partial density at the indicated ``k``-Point and return it.
"""
function compute_partial_density(basis, kpt, ψk, occupation)
    @assert length(occupation) == size(ψk, 2)

    # Build the partial density for this k-Point
    ρk_real = similar(ψk[:, 1], basis.fft_size)
    ρk_real .= 0
    for (ist, ψik) in enumerate(eachcol(ψk))
        ψik_real = G_to_r(basis, kpt, ψik)
        ρk_real .+= occupation[ist] .* abs2.(ψik_real)
    end

    # Check sanity of the density (real, positive and normalized)
    T = real(eltype(ρk_real))
    check_real(ρk_real)
    if all(occupation .> 0)
        minimum(real(ρk_real)) < 0 && @warn("Negative ρ detected",
                                            min_ρ=minimum(real(ρk_real)))
    end
    n_electrons = sum(ρk_real) * basis.model.unit_cell_volume / prod(basis.fft_size)
    if abs(n_electrons - sum(occupation)) > sqrt(eps(T))
        @warn("Mismatch in number of electrons", sum_ρ=n_electrons,
              sum_occupation=sum(occupation))
    end

    # FFT and return
    r_to_G(basis, ρk_real)
end


"""
    compute_density(basis::PlaneWaveBasis, ψ::AbstractVector, occupation::AbstractVector)

Compute the density for a wave function `ψ` discretised on the plane-wave grid `basis`,
where the individual k-Points are occupied according to `occupation`. `ψ` should
be one coefficient matrix per k-Point.
"""
function compute_density(basis::PlaneWaveBasis, ψ::AbstractVector,
                         occupation::AbstractVector)
    n_k = length(basis.kpoints)

    # Sanity checks
    @assert n_k == length(ψ)
    @assert n_k == length(occupation)
    for ik in 1:n_k
        @assert length(G_vectors(basis.kpoints[ik])) == size(ψ[ik], 1)
        @assert length(occupation[ik]) == size(ψ[ik], 2)
    end
    @assert n_k > 0

    # Allocate an accumulator for ρ in each thread
    ρaccus = [similar(ψ[1][:, 1], basis.fft_size) for ithread in 1:Threads.nthreads()]

    # TODO Better load balancing ... the workload per kpoint depends also on
    #      the number of symmetry operations. We know heuristically that the Gamma
    #      point (first k-Point) has least symmetry operations, so we will put
    #      some extra workload there if things do not break even
    kpt_per_thread = [ifelse(i <= n_k, [i], Vector{Int}()) for i in 1:Threads.nthreads()]
    if n_k >= Threads.nthreads()
        kblock = floor(Int, length(basis.kpoints) / Threads.nthreads())
        kpt_per_thread = [collect(1:length(basis.kpoints) - (Threads.nthreads() - 1) * kblock)]
        for ithread in 2:Threads.nthreads()
            push!(kpt_per_thread, kpt_per_thread[end][end] .+ collect(1:kblock))
        end
        @assert kpt_per_thread[end][end] == length(basis.kpoints)
    end

    Gs = collect(G_vectors(basis))
    Threads.@threads for (ikpts, ρaccu) in collect(zip(kpt_per_thread, ρaccus))
        ρaccu .= 0
        for ik in ikpts
            ρ_k = compute_partial_density(basis, basis.kpoints[ik], ψ[ik], occupation[ik])
            _symmetrize_ρ!(ρaccu, ρ_k, basis, basis.ksymops[ik], Gs)
        end
    end

    count = sum(length(basis.ksymops[ik]) for ik in 1:length(basis.kpoints))
    #corrigated normalization for collinear spin
    n_spin=number_of_spins(basis.model)
    count=count/n_spin
    from_fourier(basis, sum(ρaccus) / count; assume_real=true)
end

#computing spin densities, total density and magnetic density
function compute_spin_densities(basis::PlaneWaveBasis, ψ::AbstractVector,
                         occupation::AbstractVector)

    n_spin=number_of_spins(basis.model)
    n_k = floor(Int, length(basis.kpoints)/n_spin)
    println("number of spins: $n_spin" )
    # Sanity checks
    @assert n_k == length(ψ)/n_spin
    @assert n_k == length(occupation)/n_spin
    #for ik in 1:n_k
    #    @assert length(G_vectors(basis.kpoints[n_spin*ik])) == size(ψ[n_spin*ik], 1)
    #    @assert length(occupation[n_spin*ik]) == size(ψ[n_spin*ik], 2)
    #end
    @assert n_k > 0


    kpt_per_thread = [ifelse(i <= n_k, [i], Vector{Int}()) for i in 1:Threads.nthreads()]
    if n_k >= Threads.nthreads()
        kblock = floor(Int, n_k / Threads.nthreads())
        kpt_per_thread = [collect(1:n_k - (Threads.nthreads() - 1) * kblock)]
        for ithread in 2:Threads.nthreads()
            push!(kpt_per_thread, kpt_per_thread[end][end] .+ collect(1:kblock))
        end
        @assert kpt_per_thread[end][end] == n_k
    end

    Gs = collect(G_vectors(basis))
    ρspin=0.
    if n_spin == 2   
        # Allocate an accumulator for ρ in each thread
        ρaccus_α = [similar(ψ[1][:, 1], basis.fft_size) for ithread in 1:Threads.nthreads()]
        ρaccus_β = [similar(ψ[1][:, 1], basis.fft_size) for ithread in 1:Threads.nthreads()]
        Threads.@threads for (ikpts, ρaccu) in collect(zip(kpt_per_thread, ρaccus_α))
            ρaccu .= 0
            for ik in ikpts
                ρα_k = compute_partial_density(basis, basis.kpoints[floor(Int,2*ik-1)], ψ[floor(Int,2*ik-1)], occupation[floor(Int,2*ik-1)])
                _symmetrize_ρ!(ρaccu, ρα_k, basis, basis.ksymops[floor(Int,2*ik-1)], Gs)
            end
        end
        Threads.@threads for (ikpts, ρaccu) in collect(zip(kpt_per_thread, ρaccus_β))
            ρaccu .= 0
            for ik in ikpts
                ρβ_k = compute_partial_density(basis, basis.kpoints[floor(Int,2*ik)], ψ[floor(Int,2*ik)], occupation[floor(Int,2*ik)])
                _symmetrize_ρ!(ρaccu, ρβ_k, basis, basis.ksymops[floor(Int,2*ik)], Gs)
            end
        end

        ρ_magnetic=ρaccus_α-ρaccus_β
        ρ_total=ρaccus_α+ρaccus_β
        count = sum(length(basis.ksymops[ik]) for ik in 1:length(basis.kpoints))
        ρtot=from_fourier(basis, sum(ρ_total) / (count/n_spin); assume_real=true)
	ρspinaccus=collect(Iterators.flatten(zip(ρaccus_α,ρaccus_β)))
        ρspin=from_fourier(basis, sum(ρspinaccus) / (count/n_spin); assume_real=true)
        #ρdiff=from_fourier(basis, sum(ρ_magnetic) / (count/n_spin); assume_real=true)
        #ρα=from_fourier(basis, sum(ρaccus_α) / (count/n_spin); assume_real=true)
        #ρβ=from_fourier(basis, sum(ρaccus_β) / (count/n_spin); assume_real=true)
	#ρspin=collect(Iterators.flatten(zip(ρα,ρβ)))
        (ρtot,ρspin)			      
    else
        ρaccus = [similar(ψ[1][:, 1], basis.fft_size) for ithread in 1:Threads.nthreads()]
        Threads.@threads for (ikpts, ρaccu) in collect(zip(kpt_per_thread, ρaccus))
            ρaccu .= 0
            for ik in ikpts
                ρ_k = compute_partial_density(basis, basis.kpoints[ik], ψ[ik], occupation[ik])
                _symmetrize_ρ!(ρaccu, ρ_k, basis, basis.ksymops[ik], Gs)
            end
        end
        ρ_total=ρaccus
        count = sum(length(basis.ksymops[ik]) for ik in 1:length(basis.kpoints))
        ρtot=from_fourier(basis, sum(ρ_total) / (count/n_spin); assume_real=true)
        (ρtot)			      
        (ρtot,ρspin)			      
    end
end


# For a given kpoint, accumulates the symmetrized versions of the
# density ρin into ρout. No normalization is performed
function _symmetrize_ρ!(ρaccu, ρin, basis, ksymops, Gs)
    T = eltype(basis)
    for (S, τ) in ksymops
        invS = Mat3{Int}(inv(S))
        # Common special case, where ρin does not need to be processed
        if iszero(S - I) && iszero(τ)
            ρaccu .+= ρin
            continue
        end

        # Transform ρin -> to the partial density at S * k.
        #
        # Since the eigenfunctions of the Hamiltonian at k and Sk satisfy
        #      u_{Sk}(x) = u_{k}(S^{-1} (x - τ))
        # with Fourier transform
        #      ̂u_{Sk}(G) = e^{-i G \cdot τ} ̂u_k(S^{-1} G)
        # equivalently
        #      ̂ρ_{Sk}(G) = e^{-i G \cdot τ} ̂ρ_k(S^{-1} G)
        for (ig, G) in enumerate(Gs)
            igired = index_G_vectors(basis, invS * G)
            if igired !== nothing
                @inbounds ρaccu[ig] += cis(-2T(π) * dot(G, τ)) * ρin[igired]
            end
        end
    end  # (S, τ)
end


"""
Interpolate a function expressed in a basis `b_in` to a basis `b_out`
This interpolation uses a very basic real-space algorithm, and makes
a DWIM-y attempt to take into account the fact that b_out can be a supercell of b_in
"""
function interpolate_density(ρ_in::RealFourierArray, b_out::PlaneWaveBasis)
    ρ_out = interpolate_density(ρ_in.real, ρ_in.basis.fft_size, b_out.fft_size,
                                ρ_in.basis.model.lattice, b_out.model.lattice)
    from_real(b_out, ρ_out)
end

# TODO Specialisation for the common case lattice_out = lattice_in
function interpolate_density(ρ_in::AbstractArray, grid_in, grid_out, lattice_in, lattice_out=lattice_in)
    T = real(eltype(ρ_in))
    @assert size(ρ_in) == grid_in

    # First, build supercell, array of 3 ints
    supercell = zeros(Int, 3)
    for i = 1:3
        if norm(lattice_in[:, i]) == 0
            @assert norm(lattice_out[:, i]) == 0
            supercell[i] = 1
        else
            supercell[i] = round(Int, norm(lattice_out[:, i]) / norm(lattice_in[:, i]))
        end
        if norm(lattice_out[:, i] - supercell[i]*lattice_in[:, i]) > .3*norm(lattice_out[:, i])
            @warn "In direction $i, the output lattice is very different from the input lattice"
        end
    end

    # ρ_in represents a periodic function, on a grid 0, 1/N, ... (N-1)/N
    grid_supercell = grid_in .* supercell
    ρ_in_supercell = similar(ρ_in, (grid_supercell...))
    for i = 1:supercell[1]
        for j = 1:supercell[2]
            for k = 1:supercell[3]
                ρ_in_supercell[
                    1 + (i-1)*grid_in[1] : i*grid_in[1],
                    1 + (j-1)*grid_in[2] : j*grid_in[2],
                    1 + (k-1)*grid_in[3] : k*grid_in[3]] = ρ_in
            end
        end
    end

    # interpolate ρ_in_supercell from grid grid_supercell to grid_out
    axes_in = (range(0, 1, length=grid_supercell[i]+1)[1:end-1] for i=1:3)
    itp = interpolate(ρ_in_supercell, BSpline(Quadratic(Periodic(OnCell()))))
    sitp = scale(itp, axes_in...)
    ρ_interp = extrapolate(sitp, Periodic())
    ρ_out = similar(ρ_in, grid_out)
    for i = 1:grid_out[1]
        for j = 1:grid_out[2]
            for k = 1:grid_out[3]
                ρ_out[i, j, k] = ρ_interp((i-1)/grid_out[1],
                                          (j-1)/grid_out[2],
                                          (k-1)/grid_out[3])
            end
        end
    end

    ρ_out
end
