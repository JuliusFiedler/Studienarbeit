using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots
gr()
println("-------------------------------------------------------------------")
#
function wp(du, u, p, t)

    du[1] = u[3] + p[1]*sin(u[3])
    du[2] = u[4] + p[2]*cos(u[3])
    du[3] = p[3]*cos(u[3]) + p[4]*sin(u[1])
    du[4] = p[5]*sin(u[3])
end

function lotka(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end
p_ = Float32[1.3, 0.9, 0.8, 1.8]
tspan = (0.0f0,3.0f0)
# u0 = Float32[1,0,0,0]
u0 = Float32[0.44249296,4.6280594]
prob = ODEProblem(lotka, u0,tspan, p_)
dt = 0.1
t_solution = solve(prob, DP5(), abstol=1e-12, reltol=1e-12, saveat = dt)
println(t_solution)
plot(t_solution)
DX = Array(t_solution(t_solution.t, Val{1}))


function calc_centered_difference(x_,dt=0.1)
    x = x_'
    x_dot_ = similar(x)
    t = dt
    x_dot_[2:end-1, :] = (x[3:end, :] - x[1:end-2, :]) / (2 * t)
    x_dot_[1, :] = (-11 / 6 * x[1, :] + 3 * x[2, :]
    - 3 / 2 * x[3, :] + x[4, :] / 3) / t
    x_dot_[end, :] = (11 / 6 * x[end, :] - 3 * x[end-1, :]
    + 3 / 2 * x[end-2, :] - x[end-3, :] / 3) / t
    return x_dot_
end

x_dot = calc_centered_difference(t_solution,dt)
