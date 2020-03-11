using ForwardDiff
using ProgressMeter


"""
Compute the independent-particle susceptibility. Will blow up for large systems
"""
function compute_χ0(ham)
    basis = ham.basis
    model = basis.model
    fft_size = basis.fft_size
    @assert length(basis.kpoints) == 1
    @assert model.spin_polarisation == :none
    filled_occ = DFTK.filled_occupation(model)
    N = length(G_vectors(basis.kpoints[1]))
    @assert N < 10_000

    Hk = ham.blocks[1]
    E, V = eigen(Hermitian(Array(Hk)))
    occ, εF = DFTK.find_occupation(basis, [E])
    occ = occ[1]
    Vr = hcat(G_to_r.(Ref(basis), Ref(basis.kpoints[1]), eachcol(V))...)
    Vr = reshape(Vr, prod(fft_size), N)
    χ0 = zeros(eltype(V), prod(fft_size), prod(fft_size))
    @showprogress "Computing χ0 ..." for m = 1:N, n = 1:N
        enred = (E[n] - εF) / model.temperature
        @assert occ[n] ≈ filled_occ * Smearing.occupation(model.smearing, enred)
        if occ[n] < 1e-6 && occ[m] < 1e-6
            factor = 0.0
        elseif abs(E[m] - E[n]) < 1e-6
            factor = filled_occ * Smearing.occupation_derivative(model.smearing, enred) / model.temperature
        else
            factor = (occ[m] - occ[n])/(E[m] - E[n])
        end
        χ0 += (Vr[:, m] .* Vr[:, m]') .* (Vr[:, n] .* Vr[:, n]') * factor
    end
    χ0
end
