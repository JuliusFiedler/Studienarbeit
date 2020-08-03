cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots
using CSV
using DataFrames
# gr()

function calc_centered_difference(x_, dt=0.1)
    x = x_'
    x_dot_ = similar(x)
    t = dt
    x_dot_[2:end-1, :] = (x[3:end, :] - x[1:end-2, :]) / (2 * t)
    x_dot_[1, :] = (-11 / 6 * x[1, :] + 3 * x[2, :]
    - 3 / 2 * x[3, :] + x[4, :] / 3) / t
    x_dot_[end, :] = (11 / 6 * x[end, :] - 3 * x[end-1, :]
    + 3 / 2 * x[end-2, :] - x[end-3, :] / 3) / t
    return x_dot_'
end
function lotka(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] + β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  + δ*u[2]
end
print("---------------------------------------------------------------------")
# Define the experimental parameter
system = 1 # 1 = volterra, 2 = lorenz,
tspan = (0.0f0,3.0f0)
dt = .1
NN = false
maxiter = 30
NN_th = 0.1
if (system == 1) # volterra
    sys = lotka
    order = 2 # order soll sein: Summe aller Exponenten in jedem Monom
    u0 = Float32[0.44249296,4.6280594]
    p_ = Float32[1.3, -0.9, 0.8, -1.8]
    p_nom = Array{Float32}([0 0; p_[1] 0; 0 p_[4]; p_[2] p_[3]; 0 0; 0 0])
    p_k_nom = Array{Float32}([0 0; 0 0; 0 0; p_[2] p_[3]; 0 0; 0 0])

elseif (system == 2)#lorenz
    sys = lorenz
    order = 3
end

prob = ODEProblem(sys, u0,tspan, p_)


# Datenerzeugung-------------------------------------------------------
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = dt)
X = solution
DX = Array(solution(solution.t, Val{1}))
display(plot(X, title = "Lösung der DGL"))
display(plot(DX', label = "komische Ableitung", title = "Ableitungen"))

x_dot = calc_centered_difference(X,dt)
display(plot!(x_dot', label = "Zentraldifferenz", title = "Ableitungen"))

if (system == 1)
    DX_ = [p_[1]*(X[1,:])'+p_[2]*(X[1,:].*X[2,:])';p_[3]*(X[1,:].*X[2,:])'+p_[4]*(X[2,:])']
elseif (system ==2)

end

display(plot!(DX_', label = "exakte Ableitung", title = "Ableitungen"))

display(plot(DX'.-DX_', title = "Fehler komische Ableitung vs exakte Ableitung"))
display(plot(x_dot'.-DX_', title = "Fehler Zentraldifferenz vs exakte Ableitung"))

# CSV.write("DX.csv", DataFrame(DX')) #Quatsch
CSV.write("X.csv", DataFrame(X'))
CSV.write("DX_.csv", DataFrame(DX_')) # exakt
CSV.write("x_dot.csv", DataFrame(x_dot')) #Zentraldifferenz

max_error = 0
NN_params = []
if NN
    tsdata = Array(solution)
    # Add noise to the data
    noisy_data = tsdata + Float32(1e-5)*randn(eltype(tsdata), size(tsdata))

    ann = FastChain(FastDense(2, 32, tanh),FastDense(32, 32, tanh), FastDense(32, 2))
    p = initial_params(ann)
    function dudt2(u, p,t)
        x, y = u
        z = ann(u,p)
        [ z[1],
         z[2]]
    end
    prob_nn = ODEProblem(dudt2, u0, tspan, p)

    function predict(θ)
        Array(concrete_solve(prob_nn, Vern7(), u0, θ, saveat = X.t,
                             abstol=1e-6, reltol=1e-6,
                             sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
    end

    # No regularisation right now
    function loss(θ)
        pred = predict(θ)
        sum(abs2, noisy_data .- pred), pred # + 1e-5*sum(sum.(abs, params(ann)))
    end

    loss(p)

    const losses = []
    callback(θ,l,pred) = begin
        push!(losses, l)
        if length(losses)%50==0
            println(losses[end])
        end
        false
    end

    res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 100)
    res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)
    # print(res2.minimizer)
    # Plot the losses
    display(plot(losses, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss"))

    # Plot the data and the approximation
    NNsolution = predict(res2.minimizer)

    # Trained on noisy data vs real solution
    tsdata = Array(X)
    display(plot(solution.t, NNsolution', title= "NNsolution"))
    display(plot!(solution.t, tsdata'))
    display(plot(solution.t, tsdata'.-NNsolution', title = "Fehler in X"))
    L = ann(X,res2.minimizer)
    display(plot(x_dot', title = "Zentraldifferenz"))
    display(plot!(L',title = "NN Ableitung"))
    display(plot(DX_'-L', title = "Fehler in Ableitung"))
    max_error, = findmax(broadcast(abs, DX_'-L'))
    if max_error > NN_th
        println("max error = ", max_error)
        error("Neural Network not acurate enough!, Retrain!")
    end
    CSV.write("L.csv", DataFrame(L'))
    CSV.write("ann.csv", DataFrame(res2.minimizer'))
    NN_params = res2.minimizer
else
    NN_params_from_csv = Array{Float32}(DataFrame!(CSV.File("ann.csv")))'
    NN_params = NN_params_from_csv
    L = ann(X,NN_params_from_csv)
end
X_high_res_0_01 = Array(concrete_solve(prob_nn, Vern7(), u0, NN_params, saveat = 0.01,
                     abstol=1e-6, reltol=1e-6,
                     sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
L_high_res_0_01 = ann(X_high_res_0_01, NN_params)
CSV.write("X_hr_0_01.csv", DataFrame(X_high_res_0_01'))
CSV.write("L_hr_0_01.csv", DataFrame(L_high_res_0_01'))

X_high_res_0_001 = Array(concrete_solve(prob_nn, Vern7(), u0, NN_params, saveat = 0.001,
                     abstol=1e-6, reltol=1e-6,
                     sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
L_high_res_0_001 = ann(X_high_res_0_001, NN_params)
CSV.write("X_hr_0_001.csv", DataFrame(X_high_res_0_001'))
CSV.write("L_hr_0_001.csv", DataFrame(L_high_res_0_001'))
# SINDy -------------------------------------------------------------
@variables u[1:2]
# Lots of polynomials

polys = Operation[1]
for i ∈ 1:order
    push!(polys, u[1]^i)
    push!(polys, u[2]^i)
    for j ∈ 1:(order-i)
        push!(polys, u[1]^i*u[2]^j)
    end
end
h = [polys...]
basis = Basis(h, u)

# Create an optimizer for the SINDY problem
opt = SR3()
# Create the thresholds which should be used in the search process
λ = (0.2:0.2)
# λ = exp10.(-6:0.1:2)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
f_target(x, w) = iszero(x[1]) ? Inf : norm(w.*x, 2)

print("Sindy nominal\n")
Ψ_nominal = SInDy(X[:, :], DX_[:, :], basis, λ, opt = opt, maxiter = maxiter, f_target = f_target, normalize = false, convergence_error = exp10(-10)) #
# println(Ψ_nominal)
print_equations(Ψ_nominal)
p̂ = parameters(Ψ_nominal)
print("parameters: ", p̂, "\n")
p_ident_nominal = Ψ_nominal.coeff

print("Sindy zentral\n")
Ψ_zentral = SInDy(X[:, :], x_dot[:, :], basis, λ, opt = opt, maxiter = maxiter, f_target = f_target, normalize = false, convergence_error = exp10(-10)) #
# println(Ψ_zentral)
print_equations(Ψ_zentral)
p̂ = parameters(Ψ_zentral)
print("parameters: ", p̂, "\n")
p_ident_zentral = Ψ_zentral.coeff

print("Sindy NN_0_1\n")
Ψ_NN_0_1 = SInDy(X[:, :], L[:, :], basis, λ, opt = opt, maxiter = maxiter, f_target = f_target, normalize = false, convergence_error = exp10(-10)) #
# println(Ψ_NN_0_1)
print_equations(Ψ_NN_0_1)
p̂ = parameters(Ψ_NN_0_1)
print("parameters: ", p̂, "\n")
p_ident_NN_0_1 = Ψ_NN_0_1.coeff

print("Sindy high_res_0_01\n")
Ψ_high_res_0_01 = SInDy(X_high_res_0_01[:, :], L_high_res_0_01[:, :], basis, λ, opt = opt, maxiter = maxiter, f_target = f_target, normalize = false, convergence_error = exp10(-10)) #
# println(Ψ_high_res_0_01)
print_equations(Ψ_high_res_0_01)
p̂ = parameters(Ψ_high_res_0_01)
print("parameters: ", p̂, "\n")
p_ident_high_res_0_01 = Ψ_high_res_0_01.coeff

print("Sindy high_res_0_001\n")
Ψ_high_res_0_001 = SInDy(X_high_res_0_001[:, :], L_high_res_0_001[:, :], basis, λ, opt = opt, maxiter = maxiter, f_target = f_target, normalize = false, convergence_error = exp10(-10)) #
# println(Ψ_high_res_0_001)
print_equations(Ψ_high_res_0_001)
p̂ = parameters(Ψ_high_res_0_001)
print("parameters: ", p̂, "\n")
p_ident_high_res_0_001 = Ψ_high_res_0_001.coeff
# Sindy so wie man es erwarten würde --------------------------
# Matrizenformat hier wie in Python / bzw wie man es erwarten würde
function sindy_naive(X, Ẋ, basis, λ = 0.2)
    λ = 0.2
    Θ = basis(X)' # Library eingesetzt
    Ẋ = Ẋ' # Ableitungen
    Ξ = inv(Θ'*Θ)*Θ'*Ẋ # initial guess
    for i ∈ (1:length(Ξ))
        Ξ[i] *= (abs(Ξ[i])<λ ? 0 : 1) # kleine coeff nullsetzen
    end
    Ξ_select = []
    for i ∈ (1:size(Ξ)[2])
        a = []
        for j ∈ (1:size(Ξ)[1])
            if !iszero(Ξ[j, i])
                append!(a, j)
            end
        end
        Θ_select = Θ[:, a]
        Ẋ_select = Ẋ[:, i]
        append!(Ξ_select, inv(Θ_select'*Θ_select)*Θ_select'*Ẋ_select)
    end
    count = 0
    for i ∈ (1:size(Ξ)[2])
        a = []
        for j ∈ (1:size(Ξ)[1])
            if !iszero(Ξ[j, i])
                count += 1
                Ξ[j, i] = Ξ_select[count]
            end
        end
    end
    return SparseIdentificationResult(Ξ, basis, 1, opt, true, Ẋ', X)
end
function sindy_naive_I(X, Ẋ, basis, λ = 0.2)
    λ = 0.2
    Θ = basis(X)' # Library eingesetzt
    Ẋ = Ẋ' # Ableitungen
    I = Diagonal(ones(eltype(Ẋ), size(Θ)[2]))
    Ξ = inv(Θ' * Θ ) * Θ' * Ẋ # initial guess
    for i ∈ (1:length(Ξ))
        Ξ[i] *= (abs(Ξ[i])<λ ? 0 : 1) # kleine coeff nullsetzen
    end
    print(Ξ)
    Ξ_select = []
    for i ∈ (1:size(Ξ)[2])
        a = []
        for j ∈ (1:size(Ξ)[1])
            if !iszero(Ξ[j, i])
                append!(a, j)
            end
        end
        Θ_select = Θ[:, a]
        Ẋ_select = Ẋ[:, i]
        I = Diagonal(ones(eltype(Ẋ), size(Θ_select)[2]))
        append!(Ξ_select, inv(Θ_select'*Θ_select +I)*Θ_select'*Ẋ_select)
    end
    count = 0
    for i ∈ (1:size(Ξ)[2])
        a = []
        for j ∈ (1:size(Ξ)[1])
            if !iszero(Ξ[j, i])
                count += 1
                Ξ[j, i] = Ξ_select[count]
            end
        end
    end
    return Ξ
    # return SparseIdentificationResult(Ξ, basis, 1, opt, true, Ẋ', X)
end
println("\n Sindy naive nominal")
Ψ_naive_nom = sindy_naive(X, DX_, basis)
print_equations(Ψ_naive_nom, show_parameter = true)
p_naive_ident_nominal = Ψ_naive_nom.coeff

println("\n Sindy naive zentral ")
Ψ_naive_zentral = sindy_naive(X, x_dot, basis)
print_equations(Ψ_naive_zentral, show_parameter = true)
p_naive_ident_zentral = Ψ_naive_zentral.coeff

println("\n Sindy naive NN 0.1s ")
Ψ_naive_NN_0_1 = sindy_naive(X, L, basis)
print_equations(Ψ_naive_NN_0_1, show_parameter = true)
p_naive_ident_NN_0_1 = Ψ_naive_NN_0_1.coeff

println("\n Sindy naive NN 0.01s ")
Ψ_naive_NN_0_01 = sindy_naive(X_high_res_0_01, L_high_res_0_01, basis)
print_equations(Ψ_naive_NN_0_01, show_parameter = true)
p_naive_ident_high_res_0_01 = Ψ_naive_NN_0_01.coeff

println("\n Sindy naive NN 0.001s ")
Ψ_naive_NN_0_001 = sindy_naive(X_high_res_0_001, L_high_res_0_001, basis)
print_equations(Ψ_naive_NN_0_001, show_parameter = true)
p_naive_ident_high_res_0_001 = Ψ_naive_NN_0_001.coeff

# Auswertung --------------------------------------------------
function calc_relative_error(p_n, p_i)
    s = 0
    i = 0
    for a ∈ (1:length(p_n))
        if p_n[a] != 0
            i += 1
            s += ((p_n[a] - p_i[a]) / p_n[a])^2
        end
    end
    return sqrt(s/i)
end

println("\n OG Sindy")
println("\n   RMS relative error Nominalableitung:")
println(calc_relative_error(p_nom, p_ident_nominal))

println("\n   RMS relative error Zentraldifferenz:")
println(calc_relative_error(p_nom, p_ident_zentral))

println("\n   RMS relative error NN Sample interval = 0.1s :")
println(calc_relative_error(p_nom, p_ident_NN_0_1))

println("\n   RMS relative error NN Sample interval = 0.01s :")
println(calc_relative_error(p_nom, p_ident_high_res_0_01))

println("\n   RMS relative error NN Sample interval = 0.001s :")
println(calc_relative_error(p_nom, p_ident_high_res_0_001))

println("\n Sindy naive")
println("\n   RMS relative error Nominalableitung:")
println(calc_relative_error(p_nom, p_naive_ident_nominal))

println("\n   RMS relative error Zentraldifferenz:")
println(calc_relative_error(p_nom, p_naive_ident_zentral))

println("\n   RMS relative error NN Sample interval = 0.1s :")
println(calc_relative_error(p_nom, p_naive_ident_NN_0_1))

println("\n   RMS relative error NN Sample interval = 0.01s :")
println(calc_relative_error(p_nom, p_naive_ident_high_res_0_01))

println("\n   RMS relative error NN Sample interval = 0.001s :")
println(calc_relative_error(p_nom, p_naive_ident_high_res_0_001))
