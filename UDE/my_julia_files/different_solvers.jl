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
using Random

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

function wp(du, u, p, t)
    g, s2 = p
    du[1] = u[3]
    du[2] = u[4]
    du[3] = -g/s2*sin(u[1])
    du[4] = 0
end
print("---------------------------------------------------------------------")
# Define the experimental parameter
system = 3  # 1 = volterra, 2 = lorenz, 3 = roessler
tspan = (0.0f0,3.0f0)
dt = .1
train = false
NN = false
maxiter = 30
NN_th = 0.1
th = 0.2
multiple_trajectories = false
prior_knowledge = false
solver = Vern7()
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
    u0 = Float64[1, 1, 1]
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
Vern7()
# Datenerzeugung-------------------------------------------------------
dt = 0.01
X_wp_DP5 = solve(ODEProblem(wp, Float32[1, 1, 0.5, 0], (0.0f0,5.0f0), Float32[9.81, 0.26890449]), DP5(), abstol=1e-12, reltol=1e-12, saveat = dt)
X_wp_Vern7 = solve(ODEProblem(wp, Float32[1, 1, 0.5, 0], (0.0f0,5.0f0), Float32[9.81, 0.26890449]), Vern7(), abstol=1e-9, reltol=1e-9, saveat = dt)

X_roessler_DP5 = solve(ODEProblem(roessler, Float64[1, 1, 1], (0.0f0,5.0f0),Float64[0.2, 0.1, 5.3]), DP5(), abstol=1e-12, reltol=1e-12, saveat = dt)
X_roessler_Vern7 = solve(ODEProblem(roessler, Float64[1, 1, 1], (0.0f0,5.0f0),Float64[0.2, 0.1, 5.3]), Vern7(), abstol=1e-12, reltol=1e-12, saveat = dt)

x_dot_wp_DP5 = calc_centered_difference(X_wp_DP5, dt)
x_dot_wp_Vern7 = calc_centered_difference(X_wp_Vern7, dt)

x_dot_roessler_DP5 = calc_centered_difference(X_roessler_DP5, dt)
x_dot_roessler_Vern7 = calc_centered_difference(X_roessler_Vern7, dt)

display(plot(X_wp_DP5, title = "X WP", label = "DP5"))
display(plot!(X_wp_Vern7, title = "X WP", label = "Vern7"))

display(plot(X_roessler_DP5, title = "X Roessler", label = "DP5"))
display(plot!(X_roessler_Vern7, title = "X Roessler", label = "Vern7"))

display(plot(x_dot_wp_DP5', title = "Zentraldifferenz WP", label = "DP5"))
display(plot!(x_dot_wp_Vern7', title = "Zentraldifferenz WP", label = "Vern7"))

display(plot(x_dot_roessler_Vern7', title = "Zentraldifferenz Roessler", label = "Vern7"))
display(plot!(x_dot_roessler_DP5', title = "Zentraldifferenz Roessler", label = "DP5"))
