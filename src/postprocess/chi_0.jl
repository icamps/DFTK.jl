using ForwardDiff
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
    N = length(basis.kpoints[1].basis)
    @assert N < 10_000
    println("Computing χ0 with N=$N")

    Hk = ham.blocks[1]
    E, V = eigen(Hermitian(Array(Hk)))
    occ, εF = DFTK.find_occupation(basis, [E])
    occ = occ[1]
    Vr = G_to_r(basis, basis.kpoints[1], V)
    Vr = reshape(Vr, prod(fft_size), N)
    χ0 = zeros(eltype(V), prod(fft_size), prod(fft_size))
    for m = 1:N
        println("$m/$N")
        for n = 1:N
            @assert occ[n] ≈ filled_occ * model.smearing((E[n] - εF) / model.temperature)
            if occ[n] < 1e-6 && occ[m] < 1e-6
                factor = 0.0
            elseif abs(E[m] - E[n]) < 1e-6
                factor = filled_occ * ForwardDiff.derivative(ε -> model.smearing((ε - εF) / model.temperature), E[n])
            else
                factor = (occ[m] - occ[n])/(E[m] - E[n])
            end
            χ0 += (Vr[:, m] .* Vr[:, m]') .* (Vr[:, n] .* Vr[:, n]') * factor
        end
    end
    χ0
end
