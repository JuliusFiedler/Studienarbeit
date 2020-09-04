# cd(@__DIR__)
# using Pkg; Pkg.activate("."); Pkg.instantiate()

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
using Random

include("MyFunctions.jl")
@time using .MyFunctions

print("---------------------------------------------------------------------")
# Define the experimental parameter
system = 1  # 1 = volterra, 2 = lorenz, 3 = roessler
tspan = (0.0f0,3.0f0)
dt = .1
train = false
NN = false
maxiter = 30
NN_th = 0.1
th = 0.2
multiple_trajectories = true
no_tr = 10
prior_knowledge = false

if (system == 1) # volterra
    sys = lotka
    order = 2 # order soll sein: Summe aller Exponenten in jedem Monom
    u0 = Float32[0.44249296,4.6280594]
    p_ = Float32[1.3, -0.9, 0.8, -1.8]
    p_nom = Array{Float32}([0.0 0.0; p_[1] 0.0; 0.0 p_[4]; 0.0 0.0; p_[2] p_[3]; 0.0 0.0])
    p_k_nom = Array{Float32}([0 0; 0 0; 0 0; 0 0; p_[2] p_[3]; 0 0])
    n = 2 # Anzahl Zustände für Konstruktion NN
    name = "volterra"
    prior_knowledge = false
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
    dt = .01
    sys = roessler
    order = 2
    u0 = Float64[1, 1, -1]
    p_ = Float64[0.2, 0.1, 5.3]
    p_nom = Array{Float64}([0.0 0.0 p_[2]; 0.0 1 0.0; -1 p_[1] 0.0; -1 0.0 -p_[3]; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
    n = 3
    name = "roessler"
    th = 0.05 # wenn parameter klein
elseif system == 4 # wagen pendel
    tspan = (0.0f0,5.0f0)
    sys = wp
    p_ = Float32[9.81, 0.26890449]
    # u0 = Float32[-3, 1, 3, 1]
    u0 = Float32[1, 1, 0.5, 0]
    order = 1
    name = "wp"
    n = 4
    dt = 0.1
    multiple_trajectories = false
    solver = DP5()
end

# Datenerzeugung-------------------------------------------------------

X, DX_, x_dot = create_data(sys, u0, tspan, p_, dt)

function prior_k(X, DX_, x_dot, system)
    if system == 1
        a = 1.3
        c = -1.8
        DX_K = similar(DX_)
        x_dot_K = similar(x_dot)
        DX_K'[:, 1] = DX_'[:, 1] - a*X'[:, 1]
        DX_K'[:, 2]= DX_'[:, 2] - c*X'[:, 2]
        x_dot_K'[:, 1] = x_dot'[:, 1] - a*X'[:, 1]
        x_dot_K'[:, 2]= x_dot'[:, 2] - c*X'[:, 2]
    end
    return DX_K, x_dot_K
end


if prior_knowledge
    DX_, x_dot = prior_k(X, DX_, x_dot, system)
    p_nom = p_k_nom
end


if multiple_trajectories
    X, DX_, x_dot = mul_tra!(sys, X, DX_, x_dot, u0, tspan, p_, dt, no_tr)
end

display(plot(DX_', label = "exakte Ableitung", title = "Ableitungen"))
display(plot!(x_dot', label = "Zentraldifferenz", title = "Ableitungen"))

display(plot(x_dot'.-DX_', title = "Absoluter Fehler Zentraldifferenz vs exakte Ableitung"))
display(plot(broadcast(abs, (x_dot'.-DX_')./DX_'), title = "relativer Fehler Zentraldifferenz vs exakte Ableitung"))

if false
    CSV.write(string("X_",name,".csv"), DataFrame(X'))
    CSV.write(string("DX__",name,".csv"), DataFrame(DX_')) # exakt
    CSV.write(string("x_dot_",name,".csv"), DataFrame(x_dot')) #Zentraldifferenz
end

max_error = 0
NN_params = []
ann = FastChain(FastDense(n, 32, tanh),FastDense(32, 32, tanh), FastDense(32, n))

tsdata = Array(X)
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
elseif system == 4
    function dudt2(u, p, t)
        z = ann(u,p)
        [z[1],
         z[2],
         z[3],
         z[4]]
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
        display(plot(X.t, NNsolution', title= "NNsolution"))
        display(plot!(X.t, tsdata', title = "solution"))
        display(plot(X.t, tsdata'.-NNsolution', title = "Fehler in X"))
        L = ann(X,res2.minimizer)
        display(plot(DX_', title = "exakte Ableitung"))
        display(plot!(L',title = "NN Ableitung"))
        max_error, = findmax(broadcast(abs, (DX_'-L')./DX_'))
        if max_error > NN_th
            println("max error = ", max_error)
            @warn("Neural Network not acurate enough!, Retrain!")
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

    X_high_res_0_01 = Array(concrete_solve(prob_nn, Vern7(), u0, NN_params, saveat = dt/10,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
    L_high_res_0_01 = ann(X_high_res_0_01, NN_params)
    CSV.write(string("X_hr_0_01_",name,".csv"), DataFrame(X_high_res_0_01'))
    CSV.write(string("L_hr_0_01_",name,".csv"), DataFrame(L_high_res_0_01'))

    X_high_res_0_001 = Array(concrete_solve(prob_nn, Vern7(), u0, NN_params, saveat = dt/100,
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
if system == 4
    h= [sin.(u)..., cos.(u)..., polys...]
end
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
@time Ψ_nominal = SInDy(X[:, :], DX_[:, :], basis, λ, opt = opt, maxiter = maxiter, normalize = false, convergence_error = exp10(-10)) #
# println(Ψ_nominal)
print_equations(Ψ_nominal)
p̂ = parameters(Ψ_nominal)
print("parameters: ", p̂, "\n")
p_ident_nominal = Ψ_nominal.coeff

print("Sindy zentral\n")
Ψ_zentral = SInDy(X[:, :], x_dot[:, :], basis, λ, opt = opt, maxiter = maxiter, normalize = false, convergence_error = exp10(-10)) #
# println(Ψ_zentral)
print_equations(Ψ_zentral)
p̂ = parameters(Ψ_zentral)
print("parameters: ", p̂, "\n")
p_ident_zentral = Ψ_zentral.coeff

if NN
    print("Sindy NN_0_1\n")
    Ψ_NN_0_1 = SInDy(X[:, :], L[:, :], basis, λ, opt = opt, maxiter = maxiter, normalize = false, convergence_error = exp10(-10)) #
    # println(Ψ_NN_0_1)
    print_equations(Ψ_NN_0_1)
    p̂ = parameters(Ψ_NN_0_1)
    print("parameters: ", p̂, "\n")
    p_ident_NN_0_1 = Ψ_NN_0_1.coeff

    print("Sindy high_res_0_01\n")
    Ψ_high_res_0_01 = SInDy(X_high_res_0_01[:, :], L_high_res_0_01[:, :], basis, λ, opt = opt, maxiter = maxiter, normalize = false, convergence_error = exp10(-10)) #
    # println(Ψ_high_res_0_01)
    print_equations(Ψ_high_res_0_01)
    p̂ = parameters(Ψ_high_res_0_01)
    print("parameters: ", p̂, "\n")
    p_ident_high_res_0_01 = Ψ_high_res_0_01.coeff

    print("Sindy high_res_0_001\n")
    Ψ_high_res_0_001 = SInDy(X_high_res_0_001[:, :], L_high_res_0_001[:, :], basis, λ, opt = opt, maxiter = maxiter, normalize = false, convergence_error = exp10(-10)) #
    # println(Ψ_high_res_0_001)
    print_equations(Ψ_high_res_0_001)
    p̂ = parameters(Ψ_high_res_0_001)
    print("parameters: ", p̂, "\n")
    p_ident_high_res_0_001 = Ψ_high_res_0_001.coeff
end


# Sindy so wie man es erwarten würde --------------------------
# Matrizenformat hier wie in Python / bzw wie man es erwarten würde

println("System: ", name)
println("\n Sindy naive nominal")
@time p_naive_ident_nominal = sindy_naive(X, DX_, basis)

println("\n Sindy naive zentral ")
p_naive_ident_zentral = sindy_naive(X, x_dot, basis)

if NN
    println("\n Sindy naive NN 0.1s ")
    p_naive_ident_NN_0_1 = sindy_naive(X, L, basis)

    println("\n Sindy naive NN 0.01s ")
    p_naive_ident_high_res_0_01 = sindy_naive(X_high_res_0_01, L_high_res_0_01, basis)

    println("\n Sindy naive NN 0.001s ")
    p_naive_ident_high_res_0_001 = sindy_naive(X_high_res_0_001, L_high_res_0_001, basis)
end
# Auswertung --------------------------------------------------
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


function export_to_csv(l = 5)

    p_ident =   [p_ident_nominal, p_ident_zentral, p_ident_NN_0_1, p_ident_high_res_0_01, p_ident_high_res_0_001,
                p_naive_ident_nominal, p_naive_ident_zentral, p_naive_ident_NN_0_1, p_naive_ident_high_res_0_01, p_naive_ident_high_res_0_001]
    row_DiffEq_abs = Array{String}(undef, 1, l+1)
    row_DiffEq_rel = Array{String}(undef, 1, l+1)
    row_naive_abs = Array{String}(undef, 1, l+1)
    row_naive_rel = Array{String}(undef, 1, l+1)
    row_DiffEq_abs[1] = "DiffEq abs. Fehler"
    row_DiffEq_rel[1] = "DiffEq rel. Fehler"
    row_naive_abs[1] = "SINDy naiv abs. Fehler"
    row_naive_rel[1] = "SINDy naiv rel. Fehler"
    for i ∈ (1:l)
        row_DiffEq_abs[i+1] = string(calc_param_ident_error(p_nom, p_ident[i])[1])
        row_DiffEq_rel[i+1] = string(calc_param_ident_error(p_nom, p_ident[i])[2])
        row_naive_abs[i+1] = string(calc_param_ident_error(p_nom, p_ident[i+5])[1])
        row_naive_rel[i+1] = string(calc_param_ident_error(p_nom, p_ident[i+5])[2])
    end
    rows = [row_DiffEq_abs;
            row_DiffEq_rel;
            row_naive_abs;
            row_naive_rel]
    folder = "C:/Users/Julius/Documents/Studium_Elektrotechnik/Studienarbeit/github/Studienarbeit/Latex/RST-DiplomMasterStud-Arbeit/images/"
    CSV.write(string(folder, "errors_", name, ".csv"), DataFrame(rows), append= true)
end
