using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots
gr()
println("-------------------------------------------------------------------")
#
# function wp(du, u, p, t)
#
#     du[1] = u[3] + p[1]*sin(u[3])
#     du[2] = u[4] + p[2]*cos(u[3])
#     du[3] = p[3]*cos(u[3]) + p[4]*sin(u[1])
#     du[4] = p[5]*sin(u[3])
# end
#
# tspan = (0.0f0,3.0f0)
# u0 = Float32[1,0,0,0]
# prob = ODEProblem(wp, u0,tspan, a_prob.p)
# t_solution = solve(prob, DP5(), abstol=1e-12, reltol=1e-12, saveat = 0.1)
# println(t_solution)
# plot(t_solution)
f_target(x, w) = iszero(x[1]) ? Inf : norm(w.*x, 2)
