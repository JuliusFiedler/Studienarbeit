cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

# Single experiment, move to ensemble further on
# Some good parameter values are stored as comments right now
# because this is really good practice

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots
gr()

print("---------------------------------------------------------------------")
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
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 0.1)

display(scatter(solution, alpha = 0.25))
display(plot!(solution, alpha = 0.5))

# Ideal data
tsdata = Array(solution)
# Add noise to the data
noisy_data = tsdata + Float32(1e-5)*randn(eltype(tsdata), size(tsdata))

display(plot(abs.(tsdata-noisy_data)'))

# Define the neueral network which learns L(x, y, y(t-τ))
# Actually, we do not care about overfitting right now, since we want to
# extract the derivative information without numerical differentiation.
ann = FastChain(FastDense(2, 32, tanh),FastDense(32, 32, tanh), FastDense(32, 2))
p = initial_params(ann)

function dudt_(u, p,t)
    x, y = u
    z = ann(u,p)
    [p_[1]*x + z[1],
    -p_[4]*y + z[2]]
end


prob_nn = ODEProblem(dudt_,u0, tspan, p)
s = concrete_solve(prob_nn, Tsit5(), u0, p, saveat = solution.t)

display(plot(solution))
display(plot!(s))

function predict(θ)
    Array(concrete_solve(prob_nn, Vern7(), u0, θ, saveat = solution.t,
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
display(plot(solution.t, NNsolution', title= "NNsolution"))
display(plot!(solution.t, tsdata'))
display(plot(solution.t, tsdata'.-NNsolution'))

# Collect the state trajectory and the derivatives
X = noisy_data
# Ideal derivatives

DX = Array(solution(solution.t, Val{1})) #- [p[1]*(X[1,:])';  -p[4]*(X[2,:])']

prob_nn2 = ODEProblem(dudt_,u0, tspan, res2.minimizer)
_sol = solve(prob_nn2, Tsit5())
DX_ = Array(_sol(solution.t, Val{1}))

# The learned derivatives
display(plot(DX', title = "learned derivatives"))
display(plot!(DX_'))

# Ideal data
L = [-p_[2]*(X[1,:].*X[2,:])';p_[3]*(X[1,:].*X[2,:])']
L̂ = ann(X,res2.minimizer)
display(scatter(L', title = "zu identifizierender Teil"))
display(plot!(L̂'))

display(scatter(abs.(L-L̂)', yaxis = :log))

# Create a Basis
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
print(polys,"\n")
# And some other stuff
h = [cos.(u)...; sin.(u)...; polys...]
basis = Basis(h, u)
# print(polys)

# Create an optimizer for the SINDY problem
opt = SR3()
# Create the thresholds which should be used in the search process
λ = exp10.(-6:0.1:2)
# print("\n",λ)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
f_target(x, w) = iszero(x[1]) ? Inf : norm(w.*x, 2)

# Test on original data and without further knowledge
Ψ = SInDy(X[:, :], DX[:, :], basis, λ, opt = opt, maxiter = 10000, f_target = f_target) # Fail
print("Test on original data and without further knowledge")
println(Ψ)
print_equations(Ψ)

# Test on ideal derivative data ( not available )
Ψ = SInDy(X[:, 5:end], L[:, 5:end], basis, λ, opt = opt, maxiter = 10000, f_target = f_target) # Succeed
print("Test on ideal derivative data ( not available ) \n")
println(Ψ)
print_equations(Ψ)
p̂ = parameters(Ψ)
print("parameters: ", p̂, "\n")

# Test on uode derivative data
Ψ = SInDy(X[:, 2:end], L̂[:, 2:end], basis, λ,  opt = opt, maxiter = 10000, normalize = true, denoise = true, f_target = f_target) # Succeed
print("Test on uode derivative data \n")
println(Ψ)
print_equations(Ψ)
p̂ = parameters(Ψ)
print("parameters: ", p̂, "\n")
# The parameters are a bit off, so we reiterate another sindy term to get closer to the ground truth

# Create function
unknown_sys = ODESystem(Ψ)
unknown_eq = ODEFunction(unknown_sys)
# Just the equations
b = Basis((u, p, t)->unknown_eq(u, [1.; 1.], t), u)
# Retune for better parameters -> we could also use DiffEqFlux or other parameter estimation tools here.
Ψf = SInDy(noisy_data[:, 2:end], L̂[:, 2:end], b, opt = SR3(0.01), maxiter = 100, convergence_error = 1e-18) # Succeed
print("Retune for better parameters \n")
println(Ψf)
p̂ = parameters(Ψf)
print("parameters: ", p̂, "\n")

# Create function
unknown_sys = ODESystem(Ψf)
unknown_eq = ODEFunction(unknown_sys)

# Build a ODE for the estimated system
function approx(du, u, p, t)
    # Add SInDy Term
    α, δ, β, γ = p
    z = unknown_eq(u, [β; γ], t)
    du[1] = α*u[1] + z[1]
    du[2] = -δ*u[2] + z[2]
end

# Create the approximated problem and solution
ps = [p_[[1,4]]; p̂]
print("approximated problem parameters", ps, "\n")
a_prob = ODEProblem(approx, u0, tspan, ps)
a_solution = solve(a_prob, Tsit5(), saveat = 0.1)

# Plot+#
# print(solution, "\n")
# print(a_solution,"\n")
# print(solution[1,:], solution[2,:])
display(plot(solution))
display(plot!(a_solution, linestyle= :dash))

# difference
display(plot(solution[1,:].-a_solution[1,:],label="x", title="Differenz in ersten 3 sec"))
display(plot!(solution[2,:].-a_solution[2,:],label="y"))

# Look at long term prediction
t_long = (0.0, 50.0)
a_prob = ODEProblem(approx, u0, t_long, ps)
a_solution = solve(a_prob, Tsit5()) # Using higher tolerances here results in exit of julia
display(plot(a_solution))

prob_true2 = ODEProblem(lotka, u0, t_long, p_)
solution_long = solve(prob_true2, Tsit5(), saveat = a_solution.t)
display(plot!(solution_long))

display(plot(solution_long[1,:].-a_solution[1,:],label="x", title="Differenz long"))
display(plot!(solution_long[2,:].-a_solution[2,:],label="y"))

# using JLD2
# @save "knowledge_enhanced_NN.jld2" solution unknown_sys a_solution NNsolution ann solution_long X L L̂
# @load "knowledge_enhanced_NN.jld2" solution unknown_sys a_solution NNsolution ann solution_long X L L̂

p1 = plot(0.1:0.1:tspan[end],abs.(Array(solution)[:,2:end] .- NNsolution[:,2:end])' .+ eps(Float32),
          lw = 3, yaxis = :log, title = "Timeseries of UODE Error (solution- NNsolution)",
          color = [3 :orange], xlabel = "t",
          label = ["x(t)" "y(t)"],
          titlefont = "Helvetica", legendfont = "Helvetica",
          legend = :topright)
display(p1)
# Plot L₂
p2 = plot(X[1,:], X[2,:], L[2,:], lw = 3,
     title = "Neural Network Fit of U2(t)", color = 3,
     label = "Neural Network", xaxis = "x", yaxis="y",
     titlefont = "Helvetica", legendfont = "Helvetica",
     legend = :bottomright)
display(plot!(X[1,:], X[2,:], L̂[2,:], lw = 3, label = "True Missing Term", color=:orange))

c1 = 3 # RGBA(174/255,192/255,201/255,1) # Maroon
c2 = :orange # RGBA(132/255,159/255,173/255,1) # Red
c3 = :blue # RGBA(255/255,90/255,0,1) # Orange
c4 = :purple # RGBA(153/255,50/255,204/255,1) # Purple

p3 = scatter(solution, color = [c1 c2], label = ["x data" "y data"],
             title = "Extrapolated Fit From Short Training Data",
             titlefont = "Helvetica", legendfont = "Helvetica",
             markersize = 5)

display(plot!(p3,solution_long, color = [c1 c2], linestyle = :dot, lw=5, label = ["True x(t)" "True y(t)"]))
display(plot!(p3,a_solution, color = [c3 c4], lw=1, label = ["Estimated x(t)" "Estimated y(t)"]))
display(plot!(p3,[2.99,3.01],[0.0,maximum(hcat(Array(solution),Array(a_solution)))],lw=2,color=:black, legend = :topleft))
annotate!([(1.5,9,text("Training \nData", 10, :center, :top, :black, "Helvetica"))])
l = @layout [grid(1,2)
             grid(1,1)]
display(plot(p1,p2,p3,layout = l))

savefig("sindy_extrapolation.pdf")
