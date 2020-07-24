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
gr()

print("---------------------------------------------------------------------")
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
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

# Define the experimental parameter
tspan = (0.0f0,3.0f0)
u0 = Float32[0.44249296,4.6280594]
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka, u0,tspan, p_)
dt = .1
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = dt)

X = solution
DX = Array(solution(solution.t, Val{1}))
display(plot(X, title = "Lösung der DGL"))
display(plot(DX', label = "komische Ableitung", title = "Ableitungen"))

x_dot = calc_centered_difference(X,dt)
display(plot!(x_dot', label = "Zentraldifferenz", title = "Ableitungen"))

DX_ = [p_[1]*(X[1,:])'-p_[2]*(X[1,:].*X[2,:])';p_[3]*(X[1,:].*X[2,:])'-p_[4]*(X[2,:])']
display(plot!(DX_', label = "exakte Ableitung", title = "Ableitungen"))

display(plot(DX'.-DX_', title = "Fehler komische Ableitung vs exakte Ableitung"))
display(plot(x_dot'.-DX_', title = "Fehler Zentraldifferenz vs exakte Ableitung"))

CSV.write("DX.csv", DataFrame(DX'))
CSV.write("X.csv", DataFrame(X'))
CSV.write("DX_.csv", DataFrame(DX_'))
CSV.write("x_dot.csv", DataFrame(x_dot'))

@variables u[1:2]
# Lots of polynomials
order = 2 # order soll sein: Summe aller Exponenten in jedem Monom
polys = Operation[1]
for i ∈ 1:order
    push!(polys, u[1]^i)
    push!(polys, u[2]^i)
    for j ∈ 1:(order-i)
        # if i != j
            # push!(polys, u[1]^i*u[2]^j)
            # push!(polys, u[2]^i*u[1]^i)
        push!(polys, u[1]^i*u[2]^j)
            # push!(polys, u[2]^i*u[1]^i)
            # push!(polys, u[1]^j*u[2]^i)
        #end
    end
end
# print(polys,"\n")
# And some other stuff
# h = [cos.(u)...; sin.(u)...; polys...]
h = [polys...]
basis = Basis(h, u)
# print(polys)

# Create an optimizer for the SINDY problem
opt = SR3()
# Create the thresholds which should be used in the search process
λ = (0.2:0.2)
# λ = exp10.(-6:0.1:2)
# print("\n",λ)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
f_target(x, w) = iszero(x[1]) ? Inf : norm(w.*x, 2)
print("Sindy\n")
Ψ = SInDy(X[:, :], x_dot[:, :], basis, λ, opt = opt, maxiter = 30, f_target = f_target, normalize = false, convergence_error = exp10(-10)) #
print("Test on original data and without further knowledge")
println(Ψ)
print_equations(Ψ)
p̂ = parameters(Ψ)
print("parameters: ", p̂, "\n")
