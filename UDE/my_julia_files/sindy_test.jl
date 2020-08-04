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
using Sundials
using DiffEqBase
using Combinatorics

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

function lorenz(du, u, p, t)
    α, β, γ = p
    du[1] = α * (-u[1] + u[2])
    du[2] = β * u[1] - u[2] - u[1] * u[3]
    du[3] = -γ * u[3] + u[1] * u[2]
end

function roessler(du, u, p, t)
    α, β, γ = p
    du[1] = -u[2] - u[3]
    du[2] = u[1] + α * u[2]
    du[3] = β + u[1] * u[3] - γ * u[3]
end
print("---------------------------------------------------------------------")
# Define the experimental parameter
system = 3 # 1 = volterra, 2 = lorenz, 3 = roessler
tspan = (0.0f0,3.0f0)
dt = .1
train = false
NN = false
maxiter = 30
NN_th = 0.1
th = 0.2
if (system == 1) # volterra
    sys = lotka
    order = 2 # order soll sein: Summe aller Exponenten in jedem Monom
    u0 = Float32[0.44249296,4.6280594]
    p_ = Float32[1.3, -0.9, 0.8, -1.8]
    p_nom = Array{Float32}([0 0; p_[1] 0; 0 p_[4]; p_[2] p_[3]; 0 0; 0 0])
    p_k_nom = Array{Float32}([0 0; 0 0; 0 0; p_[2] p_[3]; 0 0; 0 0])
    n = 2 # Anzahl Zustände für Konstruktion NN
    name = "volterra"
elseif (system == 2)#lorenz
    tspan = (0.0, 3.0)
    dt = 0.01 # 0.01 ok
    sys = lorenz
    order = 2
    u0 = Float64[-8, 8, 27]
    p_ = Float64[10, 28, 8/3]
    p_nom = Array{Float64}([0.0 0.0 0.0; -p_[1] p_[2] 0.0; p_[1] -1 0.0; 0.0 0.0 -p_[3]; 0.0 0.0 0.0; 0.0 0.0 1; 0.0 -1 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
    n = 3
    name = "lorenz"
elseif (system == 3)#roessler
    tspan = (0.0, 3.0)
    dt = 0.01
    sys = roessler
    order = 2
    u0 = Float64[1, 1, 1]
    p_ = Float64[0.2, 0.1, 5.3]
    p_nom = Array{Float64}([0.0 0.0 p_[2]; 0.0 1 0.0; -1 p_[1] 0.0; -1 0.0 -p_[3]; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
    n = 3
    name = "roessler"
    th = 0.05 # wenn parameter klein
end

# Datenerzeugung-------------------------------------------------------
prob = ODEProblem(sys, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = dt)
X = solution

if (system == 1)
    DX_ =   [p_[1]*(X[1,:])'+p_[2]*(X[1,:].*X[2,:])';
            p_[3]*(X[1,:].*X[2,:])'+p_[4]*(X[2,:])']
elseif (system == 2)
    DX_ =   [p_[1] * (-(X[1,:])' + (X[2,:])');
            p_[2] * (X[1,:])' - (X[2,:])' - (X[1,:])' .* (X[3,:])';
            -p_[3] * (X[3,:])' + (X[1,:])' .* (X[2,:])']
elseif (system == 3)
    DX_ =   [ -(X[2,:])' - (X[3,:])';
            (X[1,:])' + p_[1] * (X[2,:])';
            p_[2] .+ (X[1,:])' .* (X[3,:])' - p_[3] * (X[3,:])']
end

x_dot = calc_centered_difference(X,dt)

display(plot(DX_', label = "exakte Ableitung", title = "Ableitungen"))
display(plot!(x_dot', label = "Zentraldifferenz", title = "Ableitungen"))

display(plot(x_dot'.-DX_', title = "Absoluter Fehler Zentraldifferenz vs exakte Ableitung"))
display(plot(broadcast(abs, (x_dot'.-DX_')./DX_'), title = "relativer Fehler Zentraldifferenz vs exakte Ableitung"))

CSV.write(string("X_",name,".csv"), DataFrame(X'))
CSV.write(string("DX__",name,".csv"), DataFrame(DX_')) # exakt
CSV.write(string("x_dot_",name,".csv"), DataFrame(x_dot')) #Zentraldifferenz

max_error = 0
NN_params = []
ann = FastChain(FastDense(n, 32, tanh),FastDense(32, 32, tanh), FastDense(32, n))

tsdata = Array(solution)
# Add noise to the data
noisy_data = tsdata + Float32(1e-5)*randn(eltype(tsdata), size(tsdata))

p = initial_params(ann)
if system == 1
    function dudt2(u, p, t)
        z = ann(u,p)
        [z[1],
         z[2]]
    end
elseif system ∈ (2:3)
    function dudt2(u, p, t)
        z = ann(u,p)
        [z[1],
         z[2],
         z[3]]
    end
end

prob_nn = ODEProblem(dudt2, u0, tspan, p)

if NN
    if train
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
        res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 1000)
        # Plot the losses
        display(plot(losses, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss"))

        # Plot the data and the approximation
        NNsolution = predict(res2.minimizer)

        # Trained on noisy data vs real solution
        tsdata = Array(X)
        display(plot(solution.t, NNsolution', title= "NNsolution"))
        display(plot!(solution.t, tsdata', title = "solution"))
        display(plot(solution.t, tsdata'.-NNsolution', title = "Fehler in X"))
        L = ann(X,res2.minimizer)
        display(plot(DX_', title = "exakte Ableitung"))
        display(plot!(L',title = "NN Ableitung"))
        max_error, = findmax(broadcast(abs, (DX_'-L')./DX_'))
        if max_error > NN_th
            println("max error = ", max_error)
            # error("Neural Network not acurate enough!, Retrain!")
        end
        CSV.write(string("L_",name,".csv"), DataFrame(L'))
        CSV.write(string("ann_",name,".csv"), DataFrame(res2.minimizer'))
        NN_params = res2.minimizer
    else
        NN_params_from_csv = Array{Float32}(DataFrame!(CSV.File(string("ann_",name,".csv"))))'
        NN_params = NN_params_from_csv
        L = ann(X, NN_params_from_csv)
        max_error, = findmax(broadcast(abs, (DX_'-L')./DX_'))
    end

    println("\nmaximaler relativer Fehler der Ableitung des NN: ", max_error)
    display(plot(DX_'-L', title = "Fehler in Ableitung durch NN"))

    X_high_res_0_01 = Array(concrete_solve(prob_nn, Vern7(), u0, NN_params, saveat = 0.01,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
    L_high_res_0_01 = ann(X_high_res_0_01, NN_params)
    CSV.write(string("X_hr_0_01_",name,".csv"), DataFrame(X_high_res_0_01'))
    CSV.write(string("L_hr_0_01_",name,".csv"), DataFrame(L_high_res_0_01'))

    X_high_res_0_001 = Array(concrete_solve(prob_nn, Vern7(), u0, NN_params, saveat = 0.001,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
    L_high_res_0_001 = ann(X_high_res_0_001, NN_params)
    CSV.write(string("X_hr_0_001_",name,".csv"), DataFrame(X_high_res_0_001'))
    CSV.write(string("L_hr_0_001_",name,".csv"), DataFrame(L_high_res_0_001'))
end
# SINDy -------------------------------------------------------------
# Create Polynomial Library
@variables u[1:n]
polys = Operation[1]
for i ∈ (1:order)
    comb = collect(with_replacement_combinations(u,i))
    for j ∈ (1:length(comb))
        monom = comb[j][1]
        for k ∈ (2:length(comb[j]))
            monom *= comb[j][k]
        end
        push!(polys, monom)
    end
end

h = [polys...]
basis = Basis(h, u)

# Create an optimizer for the SINDY problem
opt = SR3()
# Create the thresholds which should be used in the search process
λ = (th:th)
# λ = exp10.(-6:0.1:2)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
f_target(x, w) = iszero(x[1]) ? Inf : norm(w.*x, 2)

println("System: ", name)
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

if NN
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
end
# Sindy so wie man es erwarten würde --------------------------
# Matrizenformat hier wie in Python / bzw wie man es erwarten würde
function sindy_naive(X, Ẋ, basis, λ = th)
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
    for i ∈ (1:length(Ξ))
        Ξ[i] *= (abs(Ξ[i])<λ ? 0 : 1) # kleine coeff nullsetzen
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
println("System: ", name)
println("\n Sindy naive nominal")
Ψ_naive_nom = sindy_naive(X, DX_, basis)
print_equations(Ψ_naive_nom, show_parameter = true)
p_naive_ident_nominal = Ψ_naive_nom.coeff

println("\n Sindy naive zentral ")
Ψ_naive_zentral = sindy_naive(X, x_dot, basis)
print_equations(Ψ_naive_zentral, show_parameter = true)
p_naive_ident_zentral = Ψ_naive_zentral.coeff

if NN
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
end
# Auswertung --------------------------------------------------
function calc_param_ident_error(p_n, p_i)
    s = 0
    i = 0
    t = 0
    msg = ""
    for a ∈ (1:length(p_n))
        if p_n[a] != 0
            i += 1
            s += ((p_n[a] - p_i[a]) / p_n[a])^2
        elseif p_i[a] != 0
            msg = ", allerdings zu viele Parameter identifiziert"
        end
        t += (p_n[a] - p_i[a])^2
    end
    println("   RMS absoluter Fehler: ", sqrt(t/length(p_n)))
    println("   RMS relativer Fehler: ", sqrt(s/i)," ", msg)
end
println("System: ", name)
println("\nOG Sindy")
println("\nNominalableitung:")
calc_param_ident_error(p_nom, p_ident_nominal)

println("\nZentraldifferenz:")
calc_param_ident_error(p_nom, p_ident_zentral)

if NN
    println("\nNN Sample interval = 0.1s :")
    calc_param_ident_error(p_nom, p_ident_NN_0_1)

    println("\nNN Sample interval = 0.01s :")
    calc_param_ident_error(p_nom, p_ident_high_res_0_01)

    println("\nNN Sample interval = 0.001s :")
    calc_param_ident_error(p_nom, p_ident_high_res_0_001)
end

println("\nSindy naive")
println("\nNominalableitung:")
calc_param_ident_error(p_nom, p_naive_ident_nominal)

println("\nZentraldifferenz:")
calc_param_ident_error(p_nom, p_naive_ident_zentral)

if NN
    println("\nNN Sample interval = 0.1s :")
    calc_param_ident_error(p_nom, p_naive_ident_NN_0_1)

    println("\nNN Sample interval = 0.01s :")
    calc_param_ident_error(p_nom, p_naive_ident_high_res_0_01)

    println("\nNN Sample interval = 0.001s :")
    calc_param_ident_error(p_nom, p_naive_ident_high_res_0_001)
end
