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

# include("MyFunctions.jl")
# @time using .MyFunctions: calc_centered_difference, SIR, sindy_naive, sindy_mpl,
# calc_param_ident_error, calc_param_ident_error_div_by_ev,
# wp, wp_fric, lotka, lorenz, roessler, wp_lin,
# create_data,
# mul_tra!

function mask(size, interval)
    M = zeros(size)
    for i ∈ (1:size)
        if i%interval == 0
            M[i] = true
        end
    end
    return M
end

function calc_centered_difference(x_, dt=0.1)
    x = x_'
    x_dot_ = similar(x)
    t = dt
    # in general
    x_dot_[2:end-1, :] = (x[3:end, :] - x[1:end-2, :]) / (2 * t)
    # start
    x_dot_[1, :] = (-11 / 6 * x[1, :] + 3 * x[2, :]
    - 3 / 2 * x[3, :] + x[4, :] / 3) / t
    # end
    x_dot_[end, :] = (11 / 6 * x[end, :] - 3 * x[end-1, :]
    + 3 / 2 * x[end-2, :] - x[end-3, :] / 3) / t
    return x_dot_'
end

function calc_Ξ_select(Ξ, Θ, Ẋ)
    # select relevnt columns of Θ for MKQ
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
        # append!(Ξ_select, inv(Θ_select'*Θ_select)*Θ_select'*Ẋ_select)
        append!(Ξ_select, Θ_select \ Ẋ_select)
    end
    return Ξ_select
end

function calc_Ξ_select_mpl(Ξ, Θ, Ẋ)
    # select relevnt columns of Θ for MKQ
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
        # append!(Ξ_select, inv(Θ_select'*Θ_select)*Θ_select'*Ẋ_select)
        append!(Ξ_select, pinv(Θ_select) * Ẋ_select)
    end
    return Ξ_select
end

function reshape_Ξ(Ξ, Ξ_select)
    # reshape Ξ to original size for output purpose
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
end

function SIR(Ξ, b)
    eq = Operation[]
    for i ∈ 1:size(Ξ, 2)
        eq_ = nothing
        for j ∈ 1:size(Ξ, 1)
            if !iszero(Ξ[j, i])
                if eq_ == nothing
                    eq_ = Ξ[j, i]*b.basis[j]
                else
                    eq_ += Ξ[j, i]*b.basis[j]
                end
            end
        end
        push!(eq, simplify(eq_))
        print("f", i, " = ", simplify(eq_), "\n")
    end
    return eq
end

function sindy_naive(X, Ẋ, basis, λ = 0.2, maxiter = 30)
    Θ = basis(X)' # Library eingesetzt
    Ẋ = Ẋ' # Ableitungen
    # Ξ = inv(Θ'*Θ)*Θ'*Ẋ # initial guess
    Ξ = Θ \ Ẋ
    for i ∈ (1:length(Ξ))
        Ξ[i] *= (abs(Ξ[i])<λ ? 0 : 1) # kleine coeff nullsetzen
    end
    iters = 0
    for k ∈ 1:maxiter
        Ξ_select = calc_Ξ_select(Ξ, Θ, Ẋ)
        Ξ = reshape_Ξ(Ξ, Ξ_select)

        done = true
        for i ∈ (1:length(Ξ))
            if Ξ[i] != 0 && abs(Ξ[i])<λ
                Ξ[i] *=  0  # kleine coeff nullsetzen
                done = false
            end
        end
        iters += 1
        done ? break : nothing
    end
    println("converged after ", iters, " iterations")

    # return Ξ
    # @time sir = SparseIdentificationResult(Ξ, basis, 1, SR3(), true, Ẋ', X)
    eq = SIR(Ξ, basis)
    return Ξ
end

function sindy_mpl(X, Ẋ, basis, λ = 0.2, maxiter = 30)
    Θ = basis(X)' # Library eingesetzt
    Ẋ = Ẋ' # Ableitungen
    # Ξ = inv(Θ'*Θ)*Θ'*Ẋ # initial guess
    Ξ = pinv(Θ)* Ẋ
    for i ∈ (1:length(Ξ))
        Ξ[i] *= (abs(Ξ[i])<λ ? 0 : 1) # kleine coeff nullsetzen
    end
    iters = 0
    for k ∈ 1:maxiter
        Ξ_select = calc_Ξ_select_mpl(Ξ, Θ, Ẋ)
        Ξ = reshape_Ξ(Ξ, Ξ_select)

        done = true
        for i ∈ (1:length(Ξ))
            if Ξ[i] != 0 && abs(Ξ[i])<λ
                Ξ[i] *=  0  # kleine coeff nullsetzen
                done = false
            end
        end
        iters += 1
        done ? break : nothing
    end
    println("converged after ", iters, " iterations")

    # return Ξ
    # @time sir = SparseIdentificationResult(Ξ, basis, 1, SR3(), true, Ẋ', X)
    eq = SIR(Ξ, basis)
    return Ξ
end

function calc_param_ident_error(p_no, p_i)
    digits = 5
    if size(p_no) != size(p_i)
        p_n = zeros(size(p_i))
        a, b = size(p_no)
        p_n[1:a, 1:b] = p_no
    else
        p_n = p_no
    end
    s = 0
    i = 0
    msg = ""
    for a ∈ (1:length(p_n))
        if p_n[a] != 0
            s += ((p_n[a] - p_i[a]) / p_n[a])^2
            i += 1
        elseif abs(p_i[a]) <= 1 && p_i[a] != 0
            s += (p_i[a])^2
            msg = "*"
        elseif abs(p_i[a]) > 1
            s += 1
            msg = "*"
        end
    end
    rel_error = round(sqrt(s/i), digits=digits)
    println("   RMS relativer Fehler: ", rel_error, msg)
    return rel_error #return string(rel_error, msg)
end

function calc_param_ident_error_div_by_ev(p_no, p_i)
    digits = 5
    if size(p_no) != size(p_i)
        p_n = zeros(size(p_i))
        a, b = size(p_no)
        p_n[1:a, 1:b] = p_no
    else
        p_n = p_no
    end
    s = 0
    msg = ""
    for a ∈ (1:length(p_n))
        if p_n[a] != 0
            s += ((p_n[a] - p_i[a]) / p_n[a])^2
        elseif p_i[a] != 0
            s += ((p_n[a] - p_i[a]) / p_i[a])^2
            msg = "*"
        end
    end
    rel_error = round(sqrt(s/length(p_n)), digits=digits)
    println("   RMS relativer Fehler: ", rel_error, msg)
    return string(rel_error, msg)
end


function R(u)
    d1 = 0.1
    d2 = 0.3
    return d1 .*u + d2 .*tanh.(u)
end

function wp_fric(du, u, p, t)
    m1, m2, g, s2 = p
    du[1] = u[3]
    du[2] = u[4]
    du[3] = -((g*m1 + g*m2 + m2*u[3]^2*s2*cos(u[1]))*sin(u[1]) +(m1+m2)*R(u[3])/(m2*s2))/(s2*(m1 + m2*sin(u[1])^2))
    du[4] = (m2*(g*cos(u[1]) + u[3]^2*s2)*sin(u[1])+ R(u[3])*cos(u[1])/s2)/(m1 + m2*sin(u[1])^2)
end

function wp(du, u, p, t)
    m1, m2, g, s2 = p
    du[1] = u[3]
    du[2] = u[4]
    du[3] = -(g*m1 + g*m2 + m2*u[3]^2*s2*cos(u[1]))*sin(u[1])/(s2*(m1 + m2*sin(u[1])^2))
    du[4] = (m2*(g*cos(u[1]) + u[3]^2*s2)*sin(u[1]))/(m1 + m2*sin(u[1])^2)
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

function wp_lin(du, u, p, t)
    g, s2 = p
    du[1] = u[3]
    du[2] = u[4]
    du[3] = -g/s2*sin(u[1])
    du[4] = 0
end

function create_data(sys, u0, tspan, p_, dt = 0.1, noise = 0)
    prob = ODEProblem(sys, u0, tspan, p_)
    X = Array(solve(prob, Vern7(), abstol=1e-9, reltol=1e-9, saveat = dt))

    if noise != 0
        X_noisy = X + Float32(noise)*randn(eltype(X), size(X))
        display(plot(abs.(X-X_noisy)'))
        # X = X_noisy
    end

    if sys == lotka
        DX_ =   [p_[1]*(X[1,:])'+p_[2]*(X[1,:].*X[2,:])';
                p_[3]*(X[1,:].*X[2,:])'+p_[4]*(X[2,:])']
    elseif sys == lorenz
        DX_ =   [p_[1] * (-(X[1,:])' + (X[2,:])');
                p_[2] * (X[1,:])' - (X[2,:])' - (X[1,:])' .* (X[3,:])';
                -p_[3] * (X[3,:])' + (X[1,:])' .* (X[2,:])']
    elseif sys == roessler
        DX_ =   [ -(X[2,:])' - (X[3,:])';
                (X[1,:])' + p_[1] * (X[2,:])';
                p_[2] .+ (X[1,:])' .* (X[3,:])' - p_[3] * (X[3,:])']
    elseif sys == wp
        m1, m2, g, s2 = p_
        DX_ =  [ X[3,:]';
                X[4,:]';
                -(g .*m1 + g .*m2 .+ m2.*X[3,:]'.^2 .*s2 .*cos.(X[1,:]')).*sin.(X[1,:]') ./(s2 .*(m1 .+ m2 .*sin.(X[1,:]') .^2));
                (m2 .*(g .*cos.(X[1,:]') .+ X[3,:]' .^2 .*s2) .*sin.(X[1,:]')) ./(m1 .+ m2 .*sin.(X[1,:]') .^2)]
    elseif sys == wp_lin
        DX_ = [ X[3,:]';
                X[4,:]';
                -p_[1]/p_[2]*sin.(X[1,:])';
                Float32.(zeros(size(X[1,:])))']
    elseif sys == wp_fric
        m1, m2, g, s2 = p_
        DX_ =  [ X[3,:]';
                X[4,:]';
                -((g .*m1 + g .*m2 .+ m2.*X[3,:]'.^2 .*s2 .*cos.(X[1,:]')).*sin.(X[1,:]') .+(m1+m2).*(R(X[3,:]))' /(m2.*s2))./(s2 .*(m1 .+ m2 .*sin.(X[1,:]') .^2));
                (m2 .*(g .*cos.(X[1,:]') .+ X[3,:]' .^2 .*s2) .*sin.(X[1,:]') .+ (R(X[3,:]))'.*cos.(X[1,:]')./s2) ./(m1 .+ m2 .*sin.(X[1,:]') .^2)]
    end
    if noise != 0
        X = X_noisy
    end
    x_dot = calc_centered_difference(X, dt)
    return X, DX_, x_dot
end

function mul_tra!(sys, X, DX_, x_dot, u0, tspan, p_, dt = 0.1, no_tr = 10, noise = 0)
    X, DX_, x_dot = create_data(sys, u0, tspan, p_, dt, noise)
    min_u, _ = findmin(u0)
    max_u, _ = findmax(u0)
    len = length(u0)
    A = similar(X)
    A .= X

    for i ∈ (1:(no_tr-1))
        Random.seed!(i)
        u0_ = rand(Float32, len) .*(max_u - min_u) .+ min_u
        println(u0_)
        X_, DX__, x_dot_ = create_data(sys, u0_, tspan, p_, dt, noise)
        A_ = similar(X_)
        A_ .= X_
        A = cat(A, A_, dims = 2)
        DX_ = cat(DX_, DX__, dims = 2)
        x_dot = cat(x_dot, x_dot_, dims = 2)
    end
    X = A
    return X, DX_, x_dot
end

function import_data(name)
    X_import = Array(CSV.File(string("C:/Users/Julius/Documents/Studium_Elektrotechnik/Studienarbeit/github/Studienarbeit/UDE/my_julia_files/X_",name,".csv")) |> DataFrame)'
    DX_import = Array(CSV.File(string("C:/Users/Julius/Documents/Studium_Elektrotechnik/Studienarbeit/github/Studienarbeit/UDE/my_julia_files/DX__",name,".csv")) |> DataFrame)'
    x_dot_import = Array(CSV.File(string("C:/Users/Julius/Documents/Studium_Elektrotechnik/Studienarbeit/github/Studienarbeit/UDE/my_julia_files/x_dot_",name,".csv")) |> DataFrame)'
    return X_import, DX_import, x_dot_import
end

path = "C:/Users/Julius/Documents/Studium_Elektrotechnik/Studienarbeit/github/Studienarbeit/Latex/RST-DiplomMasterStud-Arbeit/images/"


print("---------------------------------------------------------------------")
# Define the experimental parameter
system = 1  # 1 = volterra, 2 = lorenz, 3 = roessler
tspan = (0.0f0, 2.0f0)
dt = .1
train = false
NN = false
maxiter = 30
NN_th = 0.1
th = 0.2
multiple_trajectories = false
no_tr = 10
prior_knowledge = false
noise = 0
if (system == 1) # volterra
    sys = lotka
    order = 2 # order soll sein: Summe aller Exponenten in jedem Monom
    u0 = Float32[0.44249296,4.6280594]
    p_ = Float32[1.3, -0.9, 0.8, -1.8]
    p_nom = Array{Float32}([0.0 0.0; p_[1] 0.0; 0.0 p_[4]; 0.0 0.0; p_[2] p_[3]; 0.0 0.0])
    p_k_nom = Array{Float32}([0 0; 0 0; 0 0; 0 0; p_[2] p_[3]; 0 0])
    n = 2 # Anzahl Zustände für Konstruktion NN
    name = "Volterra"
    prior_knowledge = false
elseif (system == 2) # lorenz
    tspan = (0.0, 3.0)
    dt = 0.01 # 0.01 ok
    sys = lorenz
    order = 2
    u0 = Float64[-8, 8, 27]
    p_ = Float64[10, 28, 8/3]
    p_nom = Array{Float64}([0.0 0.0 0.0; -p_[1] p_[2] 0.0; p_[1] -1 0.0; 0.0 0.0 -p_[3]; 0.0 0.0 0.0; 0.0 0.0 1; 0.0 -1 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
    n = 3
    name = "Lorenz"
elseif (system == 3) # roessler
    tspan = (0.0, 10.0)
    dt = .01
    sys = roessler
    order = 2
    u0 = Float64[1, -1, 1]
    p_ = Float64[0.2, 0.1, 5.3]
    p_nom = Array{Float64}([0.0 0.0 p_[2]; 0.0 1 0.0; -1 p_[1] 0.0; -1 0.0 -p_[3]; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
    n = 3
    name = "Roessler"
    th = 0.05 # wenn parameter klein
elseif system == 4 # wagen pendel
    tspan = (0.0f0,5.0f0)
    sys = wp_lin
    p_ = Float32[9.81, 0.26890449]
    # u0 = Float32[-3, 1, 3, 1]
    u0 = Float32[1, 1, 0.5, 0]
    order = 1
    name = "Wagen-Pendel"
    n = 4
    dt = 0.1
    multiple_trajectories = true
    solver = DP5()
end

# Datenerzeugung-------------------------------------------------------

X, DX_, x_dot = create_data(sys, u0, tspan, p_, dt, noise)

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
    X, DX_, x_dot = mul_tra!(sys, X, DX_, x_dot, u0, tspan, p_, dt, no_tr, noise)
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
    # X_test = Array(CSV.File("C:/Users/Julius/Documents/Studium_Elektrotechnik/Studienarbeit/github/Studienarbeit/UDE/my_julia_files/X_roessler.csv") |> DataFrame)'
#
# if NN
#     max_error = 0
#     NN_params = []
#     ann = FastChain(FastDense(n, 32, tanh),FastDense(32, 32, tanh), FastDense(32, n))
#
#     tsdata = Array(X)
#     # Add noise to the data
#     noisy_data = tsdata + Float32(1e-5)*randn(eltype(tsdata), size(tsdata))
#
#     p = initial_params(ann)
#     if system == 1
#         function dudt2(u, p, t)
#             z = ann(u,p)
#             [z[1],
#              z[2]]
#         end
#     elseif system ∈ (2:3)
#         function dudt2(u, p, t)
#             z = ann(u,p)
#             [z[1],
#              z[2],
#              z[3]]
#         end
#     elseif system == 4
#         function dudt2(u, p, t)
#             z = ann(u,p)
#             [z[1],
#              z[2],
#              z[3],
#              z[4]]
#         end
#     end
#
#     prob_nn = ODEProblem(dudt2, u0, tspan, p)
#
#
#     if train
#         function predict(θ)
#             Array(concrete_solve(prob_nn, Vern7(), u0, θ, saveat = X.t,
#                                  abstol=1e-6, reltol=1e-6,
#                                  sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
#         end
#
#         # No regularisation right now
#         function loss(θ)
#             pred = predict(θ)
#             sum(abs2, noisy_data .- pred), pred # + 1e-5*sum(sum.(abs, params(ann)))
#         end
#
#         loss(p)
#
#         const losses = []
#         callback(θ,l,pred) = begin
#             push!(losses, l)
#             if length(losses)%50==0
#                 println(losses[end])
#             end
#             false
#         end
#
#         res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 100)
#         res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 1000)
#         # Plot the losses
#         display(plot(losses, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss"))
#
#         # Plot the data and the approximation
#         NNsolution = predict(res2.minimizer)
#
#         # Trained on noisy data vs real solution
#         tsdata = Array(X)
#         display(plot(X.t, NNsolution', title= "NNsolution"))
#         display(plot!(X.t, tsdata', title = "solution"))
#         display(plot(X.t, tsdata'.-NNsolution', title = "Fehler in X"))
#         L = ann(X,res2.minimizer)
#         display(plot(DX_', title = "exakte Ableitung"))
#         display(plot!(L',title = "NN Ableitung"))
#         max_error, = findmax(broadcast(abs, (DX_'-L')./DX_'))
#         if max_error > NN_th
#             println("max error = ", max_error)
#             @warn("Neural Network not acurate enough!, Retrain!")
#         end
#         CSV.write(string("L_",name,".csv"), DataFrame(L'))
#         CSV.write(string("ann_",name,".csv"), DataFrame(res2.minimizer'))
#         NN_params = res2.minimizer
#     else
#         NN_params_from_csv = Array{Float32}(DataFrame!(CSV.File(string("ann_",name,".csv"))))'
#         NN_params = NN_params_from_csv
#         L = ann(X, NN_params_from_csv)
#         max_error, = findmax(broadcast(abs, (DX_'-L')./DX_'))
#     end
#
#     println("\nmaximaler relativer Fehler der Ableitung des NN: ", max_error)
#     display(plot(DX_'-L', title = "Fehler in Ableitung durch NN"))
#
#     X_high_res_0_01 = Array(concrete_solve(prob_nn, Vern7(), u0, NN_params, saveat = dt/10,
#                          abstol=1e-6, reltol=1e-6,
#                          sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
#     L_high_res_0_01 = ann(X_high_res_0_01, NN_params)
#     CSV.write(string("X_hr_0_01_",name,".csv"), DataFrame(X_high_res_0_01'))
#     CSV.write(string("L_hr_0_01_",name,".csv"), DataFrame(L_high_res_0_01'))
#
#     X_high_res_0_001 = Array(concrete_solve(prob_nn, Vern7(), u0, NN_params, saveat = dt/100,
#                          abstol=1e-6, reltol=1e-6,
#                          sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
#     L_high_res_0_001 = ann(X_high_res_0_001, NN_params)
#     CSV.write(string("X_hr_0_001_",name,".csv"), DataFrame(X_high_res_0_001'))
#     CSV.write(string("L_hr_0_001_",name,".csv"), DataFrame(L_high_res_0_001'))
# end
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
# opt = SR3()
opt = STRRidge()

# Create the thresholds which should be used in the search process
λ = (th:th)
# λ = exp10.(-6:0.1:2)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
# f_target(x, w) = iszero(x[1]) ? Inf : norm(w.*x, 2)
#
# println("System: ", name)
# print("OG Sindy nominal\n")
# @time Ψ_nominal = SInDy(X[:, :], DX_[:, :], basis, λ, opt = opt, maxiter = maxiter, normalize = false, convergence_error = exp10(-10))
# # println(Ψ_nominal)
# print_equations(Ψ_nominal)
# p̂ = parameters(Ψ_nominal)
# print("parameters: ", p̂, "\n")
# p_ident_nominal = Ψ_nominal.coeff
# calc_param_ident_error(p_nom, p_ident_nominal)
#
# print("OG Sindy zentral\n")
# @time Ψ_zentral = SInDy(X[:, :], x_dot[:, :], basis, λ, opt = opt, maxiter = maxiter, normalize = false, convergence_error = exp10(-10))
# # println(Ψ_zentral)
# print_equations(Ψ_zentral)
# p̂ = parameters(Ψ_zentral)
# print("parameters: ", p̂, "\n")
# p_ident_zentral = Ψ_zentral.coeff
# calc_param_ident_error(p_nom, p_ident_zentral)
#
# if NN
#     print("Sindy NN_0_1\n")
#     Ψ_NN_0_1 = SInDy(X[:, :], L[:, :], basis, λ, opt = opt, maxiter = maxiter, normalize = false, convergence_error = exp10(-10)) #
#     # println(Ψ_NN_0_1)
#     print_equations(Ψ_NN_0_1)
#     p̂ = parameters(Ψ_NN_0_1)
#     print("parameters: ", p̂, "\n")
#     p_ident_NN_0_1 = Ψ_NN_0_1.coeff
#
#     print("Sindy high_res_0_01\n")
#     Ψ_high_res_0_01 = SInDy(X_high_res_0_01[:, :], L_high_res_0_01[:, :], basis, λ, opt = opt, maxiter = maxiter, normalize = false, convergence_error = exp10(-10)) #
#     # println(Ψ_high_res_0_01)
#     print_equations(Ψ_high_res_0_01)
#     p̂ = parameters(Ψ_high_res_0_01)
#     print("parameters: ", p̂, "\n")
#     p_ident_high_res_0_01 = Ψ_high_res_0_01.coeff
#
#     print("Sindy high_res_0_001\n")
#     Ψ_high_res_0_001 = SInDy(X_high_res_0_001[:, :], L_high_res_0_001[:, :], basis, λ, opt = opt, maxiter = maxiter, normalize = false, convergence_error = exp10(-10)) #
#     # println(Ψ_high_res_0_001)
#     print_equations(Ψ_high_res_0_001)
#     p̂ = parameters(Ψ_high_res_0_001)
#     print("parameters: ", p̂, "\n")
#     p_ident_high_res_0_001 = Ψ_high_res_0_001.coeff
# end
#
#
# # Sindy so wie man es erwarten würde --------------------------
# # Matrizenformat hier wie in Python / bzw wie man es erwarten würde
#
# println("System: ", name)
# println("\n Sindy naive nominal")
# @time p_naive_ident_nominal = sindy_naive(X, DX_, basis)
# calc_param_ident_error(p_nom, p_naive_ident_nominal)
#
# println("\n Sindy naive zentral ")
# p_naive_ident_zentral = sindy_naive(X, x_dot, basis)
# calc_param_ident_error(p_nom, p_naive_ident_zentral)
#
# if NN
#     println("\n Sindy naive NN 0.1s ")
#     p_naive_ident_NN_0_1 = sindy_naive(X, L, basis)
#
#     println("\n Sindy naive NN 0.01s ")
#     p_naive_ident_high_res_0_01 = sindy_naive(X_high_res_0_01, L_high_res_0_01, basis)
#
#     println("\n Sindy naive NN 0.001s ")
#     p_naive_ident_high_res_0_001 = sindy_naive(X_high_res_0_001, L_high_res_0_001, basis)
# end
# # Auswertung --------------------------------------------------
# if NN
#     println("\nNN Sample interval = 0.1s :")
#     calc_param_ident_error(p_nom, p_ident_NN_0_1)
#
#     println("\nNN Sample interval = 0.01s :")
#     calc_param_ident_error(p_nom, p_ident_high_res_0_01)
#
#     println("\nNN Sample interval = 0.001s :")
#     calc_param_ident_error(p_nom, p_ident_high_res_0_001)
# end
# if NN
#     println("\nNN Sample interval = 0.1s :")
#     calc_param_ident_error(p_nom, p_naive_ident_NN_0_1)
#
#     println("\nNN Sample interval = 0.01s :")
#     calc_param_ident_error(p_nom, p_naive_ident_high_res_0_01)
#
#     println("\nNN Sample interval = 0.001s :")
#     calc_param_ident_error(p_nom, p_naive_ident_high_res_0_001)
# end

# Testcases --------------------------------------------------
function do_sindy(X, DX_, x_dot, basis, opt, p_nom)
    println("System: ", name)
    print("OG Sindy nominal\n")
    t1 = @elapsed Ψ_nominal = SInDy(X[:, :], DX_[:, :], basis, λ, opt = opt, maxiter = maxiter, normalize = false, convergence_error = exp10(-10))
    # println(Ψ_nominal)
    # print_equations(Ψ_nominal)
    # p̂ = parameters(Ψ_nominal)
    # print("parameters: ", p̂, "\n")
    # print(Ψ_nominal.coeff)
    print(Ψ_nominal)
    err_js_nom = calc_param_ident_error(p_nom, Ψ_nominal)

    print("OG Sindy zentral\n")
    t2 = @elapsed Ψ_zentral = SInDy(X[:, :], x_dot[:, :], basis, λ, opt = opt, maxiter = maxiter, normalize = false, convergence_error = exp10(-10))
    # println(Ψ_zentral)
    # print_equations(Ψ_zentral)
    # p̂ = parameters(Ψ_zentral)
    # print("parameters: ", p̂, "\n")
    print(Ψ_zentral)
    err_js_zen = calc_param_ident_error(p_nom, Ψ_zentral)
    return [err_js_nom, err_js_zen], t1+t2
end

function acc_data(exp_arr, param_arr, basis=nothing, opt=nothing, p_nom = nothing)
    errors = []
    time = 0.0
    for i ∈ (1:size(param_arr)[1])
        # chose Parameter! #######################
        noise = param_arr[i]
        ##########################################
        X, DX_, x_dot = create_data(sys, u0, tspan, p_, dt, noise)
        if multiple_trajectories
            X, DX_, x_dot = mul_tra!(sys, X, DX_, x_dot, u0, tspan, p_, dt, no_tr, noise)
        end
        buf = copy(X')
        buf = cat(buf, DX_', x_dot', dims=2)
        exp_arr = cat(exp_arr, buf, dims = 1)
        exp_arr[convert(Int, round((i-1)/size(exp_arr)[2], RoundDown)+1), (i-1)%size(exp_arr)[2]+1] = size(X')[1]
        if basis != nothing
            e, t = do_sindy(X, DX_, x_dot, basis, opt, p_nom)
            push!(errors, e...)
            time += t
        end
    end
    return exp_arr, errors, time
end
# -------------------------------------------------------------------------
# create CSV


system = 1  # 1 = volterra, 2 = lorenz, 3 = roessler
tspan = (0.0f0, 1.0f0)
dt = .1
multiple_trajectories = true
no_tr = 5
opt =  STRRidge()
maxiter = 30
th = 0.2
noise = 0
if (system == 1) # volterra
    sys = lotka
    order = 5 # order soll sein: Summe aller Exponenten in jedem Monom
    u0 = Float32[0.44249296,4.6280594]
    p_ = Float32[1.3, -0.9, 0.8, -1.8]
    p_nom = Array{Float32}([0.0 0.0; p_[1] 0.0; 0.0 p_[4]; 0.0 0.0; p_[2] p_[3]; 0.0 0.0])
    p_k_nom = Array{Float32}([0 0; 0 0; 0 0; 0 0; p_[2] p_[3]; 0 0])
    n = 2 # Anzahl Zustände für Konstruktion NN
    name = "Volterra"
    prior_knowledge = false
elseif (system == 2) # lorenz

    # dt = 0.01 # 0.01 ok
    sys = lorenz
    order = 2
    u0 = Float64[-8, 8, 27]
    p_ = Float64[10, 28, 8/3]
    p_nom = Array{Float64}([0.0 0.0 0.0; -p_[1] p_[2] 0.0; p_[1] -1 0.0; 0.0 0.0 -p_[3]; 0.0 0.0 0.0; 0.0 0.0 1; 0.0 -1 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
    n = 3
    name = "Lorenz"
elseif (system == 3) # roessler

    # dt = .01
    sys = roessler
    order = 2
    u0 = Float64[1, -1, -1]
    p_ = Float64[0.2, 0.1, 5.3]
    p_nom = Array{Float64}([0.0 0.0 p_[2]; 0.0 1 0.0; -1 p_[1] 0.0; -1 0.0 -p_[3]; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1; 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
    n = 3
    name = "Roessler"
    th = 0.05 # wenn parameter klein
elseif system == 4 # wagen pendel

    sys = wp_lin
    p_ = Float32[9.81, 0.26890449]
    # u0 = Float32[-3, 1, 3, 1]
    u0 = Float32[1, 1, 0.5, 0]
    order = 1
    name = "Wagen-Pendel"
    n = 4
    dt = 0.1
    multiple_trajectories = false
    solver = DP5()
end
λ = (th:th)
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
    h = [sin.(u)..., cos.(u)..., polys...]
end
basis = Basis(h, u)


# poly order variation:
if false
    local_path = "C:/Users/Julius/Documents/Studium_Elektrotechnik/Studienarbeit/github/Studienarbeit/UDE/my_julia_files/"

    if multiple_trajectories
        X, DX_, x_dot = mul_tra!(sys, X, DX_, x_dot, u0, tspan, p_, dt, no_tr, noise)
    else
        X, DX_, x_dot = create_data(sys, u0, tspan, p_, dt)
    end

    CSV.write(string(local_path,"X_",name,".csv"), DataFrame(X'))
    CSV.write(string(local_path,"DX__",name,".csv"), DataFrame(DX_')) # exakt
    CSV.write(string(local_path,"x_dot_",name,".csv"), DataFrame(x_dot')) #Zentraldifferenz
    function poly_order_variation()
        orders = [ 2, 3, 4, 5 , 6, 7, 8]
        errors = []
        time = 0
        for ord ∈ orders
            @variables u[1:n]
            polys = Operation[1]
            for i ∈ (1:ord)
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
            e, t = do_sindy(X, DX_, x_dot, basis, opt, p_nom)
            push!(errors, e...)
            time += t
        end
        print(errors)
        p = errors[mask(length(orders)*2,2).==1]
        if all(p.!=0)
            display(plot(p, yaxis=:log))
        end
        av_time = round(time / (size(orders)[1]*2), digits=6)
        return errors, av_time
    end
    errors, av_time = poly_order_variation()
end
# tspan_variation:
# param_arr = [(0.0f0, 1.0f0),(0.0f0, 2.0f0),(0.0f0, 3.0f0),(0.0f0, 3.9f0),(0.0f0, 5.0f0)]
# dt_variation:
# param_arr = [0.0001,0.001,0.01,0.1,0.5,1]
# no_tr_variation:
# param_arr = [1,2,3,4,5,6,7,8,9,10,12,15,20]
# noise_variation:
param_arr = [0, 1e-6, 3e-6, 1e-5,3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1]

exp_arr = zeros(5, 3*n) # beginning of export array
# ea, errors, time = acc_data(exp_arr, param_arr)
print("\n----------------------------------------------------------------------\n")
ea, errors, time = acc_data(exp_arr, param_arr, basis, opt, p_nom)
print(errors)
p = errors[mask(length(param_arr)*2,2).==1]
if all(p.!=0)
    display(plot(p, yaxis=:log))
end
av_time = round(time / (size(param_arr)[1]*2), digits=6)

para = "noise"

# Did you check:
# - basis
# - envirenment parameters?
# - param_arr ?
# - paramter in acc_data?
# - para name for csv?

if false
    CSV.write(string(path,"Data_",name,"_",para,"_variation.csv"), DataFrame(ea))
end
if false
    X, DX_, x_dot = create_data(sys, u0, tspan, p_, dt, noise)
    if multiple_trajectories
        X, DX_, x_dot = mul_tra!(sys, X, DX_, x_dot, u0, tspan, p_, dt, no_tr, noise)
    end
end
if false
    do_sindy(X, DX_, x_dot, basis, opt, p_nom)
end


function export_to_csv(errors, times, file_identifier)
    no_elements = length(errors)
    rows = Array{Union{Nothing, Any}}(nothing, 1, no_elements+2)
    rows[1, 1] = "DiffEq"
    rows[1, 2:end-1] = errors
    rows[1, end] = av_time

    folder = "C:/Users/Julius/Documents/Studium_Elektrotechnik/Studienarbeit/github/Studienarbeit/Latex/RST-DiplomMasterStud-Arbeit/images/"
    CSV.write(string(folder, "errors_", name,"_",file_identifier, ".csv"), DataFrame(rows), append= true)
end


#               export_to_csv(errors, av_time, )
