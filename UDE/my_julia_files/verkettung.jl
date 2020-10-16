@time using OrdinaryDiffEq
@time using ModelingToolkit
@time using DataDrivenDiffEq
@time using LinearAlgebra
@time using DiffEqSensitivity, Optim
@time using DiffEqFlux, Flux
@time using Plots
@time using Combinatorics
@time using Random

include("MyFunctions.jl")
@time using .MyFunctions: calc_centered_difference, SIR, sindy_naive, sindy_mpl,
calc_param_ident_error, calc_param_ident_error_div_by_ev,
wp, wp_fric, lotka, lorenz, roessler, wp_lin,
create_data,
mul_tra!

println("------------------------------------------------------------")

maxiter = 30
p_ = Float32[3.34, 0.8512, 9.81, 0.26890449]
tspan = (0.0f0,5.0f0)
# u0 = Float32[-3, 3, 0.5 ,0.5]
u0 = Float32[-10, 10, 0.5 ,0.5]
prob = ODEProblem(wp, u0, tspan, p_)
dt = 0.001
m1, m2, g, s2 = p_
multiple_trajectories = true
no_tr = 5


X, DX_, x_dot = create_data(wp, u0, tspan, p_, dt)
Xf, DX_f, x_dotf = create_data(wp_fric, u0, tspan, p_, dt)

if multiple_trajectories
    X, DX_, x_dot = mul_tra!(wp, X, DX_, x_dot, u0, tspan, p_, dt, no_tr)
    Xf, DX_f, x_dotf = mul_tra!(wp_fric, X, DX_, x_dot, u0, tspan, p_, dt, no_tr)
end

display(plot(DX_', label = "exakte Ableitung", title = "Ableitungen"))
display(plot!(x_dot', label = "Zentraldifferenz", title = "Ableitungen"))

display(plot(x_dot'.-DX_', title = "Absoluter Fehler Zentraldifferenz vs exakte Ableitung"))
display(plot(broadcast(abs, (x_dot'.-DX_')./DX_'), title = "relativer Fehler Zentraldifferenz vs exakte Ableitung"))

# x_dot_l = calc_centered_difference(t_solution,dt)

n=4
@variables u[1:n]
# Lots of polynomials
poly_order = 2
polys = Operation[1]
for i ∈ (1:poly_order)
    comb = collect(with_replacement_combinations(u,i))
    for j ∈ (1:length(comb))
        monom = comb[j][1]
        for k ∈ (2:length(comb[j]))
            monom *= comb[j][k]
        end
        push!(polys, monom)
    end
end


function inverse(x)
    return 1 ./ x
end
function mult(x)
    temp = x[1]
    for i ∈ 2:length(x)
        temp *= x[i]
    end
    return temp
end
function add(x)
    return sum(x)
end
function frac(z, n)
    return z ./ n
end
function monome(u)
    return u
end
function sinus(u)
    return sin(u[1])
end
function cosinus(u)
    return cos(u[1])
end



elementarfuntionen = [sinus, cosinus, monome]
verkettungsarten = [mult]

function verkettung(f, v, d, fraction = false)
    h=Operation[]
    functions = f
    verk = v
    depth = d # Monome der Form (Π_depth(elem_func))
    # initial funtions / depth 1
    push!(h, sin(u[1]))
    push!(h, cos(u[1]))
    push!(h, u[3])

    # for i ∈ 1:length(functions)
    #     push!(h, functions[i].(u)...)
    # end
    # Verkettung
    z = Operation[1]
    for i ∈ (1:depth)
        comb = collect(with_replacement_combinations(h,i))
        for j ∈ (1:length(comb))
            for k ∈ 1:length(verk)
                push!(z, verk[k](comb[j]))
            end
        end
    end
    # push!(z, h...)
    # Fraction
    if fraction
        temp = Operation[]
        # comb = collect(combinations(h,2))
        comb = collect(combinations(z,1))
        for i ∈ 1:length(comb)
            # push!(temp, simplify(frac(comb[i][1], comb[i][2])))
            # push!(temp, frac(comb[i][1], comb[i][2]))
            push!(temp, frac(comb[i], m1+m2*sin(u[1])^2)...)
        end
        # push!(h, temp...)
        t = Operation[]
        push!(t, u...)
        push!(t, temp...)
        push!(temp, u...)
    end
    return t
    return temp
end

# große Library ############################################################
p_39_nom = Float32[-0.0 -0.0 -0.0 -0.0; -0.0 -0.0 -0.0 0.0; 0.0 0.0 -0.0 m2*g; -0.0 -0.0 -0.0 0.0; 0.0 0.0 0.0 0.0; -0.0 -0.0 -0.0 -0.0; 0.0 0.0 0.0 0.0; -0.0 0.0 0 -0.0; -0.0 0.0 -0.0 0.0; 0.0 -0.0 0.0 0.0; 0.0 -0.0 0 0.0; -0.0 0.0 -0.0 0.0;
0.0 -0.0 -0.0 m2*s2; 0.0 -0.0 -0.0 -0.0; 0.0 0.0 -0.0 0.0; 0.0 -0.0 0.0 -0.0; -0.0 -0.0 0.0 0.0; -0.0 -0.0 -0.0 0.0; 0.0 0.0 -0.0 0; -0.0 -0.0 0.0 -0.0; 0.0 0.0 -0.0 -0.0; 0.0 0.0 0.0 0.0; 0.0 -0.0 -0.0 -0.0; -0.0 -0.0 0.0 0; -0.0 0.0 0.0 -0.0;
 -0.0 0.0 -m2 0.0; 0.0 0.0 0.0 -0.0; -0.0 0.0 0.0 0.0; -0.0 -0.0 0.0 0.0; -0.0 -0.0 -0.0 -0.0; 0.0 0.0 -0.0 -0.0; -0.0 0.0 -0.0 0.0; -0.0 0.0 -g*(m1+m2)/s2 0.0; -0.0 0.0 -0.0 0.0; -0.0 0.0 0.0 0.0; -0.0 0.0 -0.0 -0.0; 0.0 -0.0 0.0 0.0; 1 0.0 -0.0 -0.0; -0.0 1 0.0 -0.0]
p_28_nom = Float32[-0.0 0.0 -0.0 0.0; -0.0 0.0 -0.0 0.0; 1 -0.0 0.0 -0.0; 0.0 1 -0.0 -0.0; 0.0 -0.0 0.0 -0.0; -0.0 0.0 -g*(m1+m2)/s2 -0.0; -0.0 -0.0 0.0 -0.0; 0.0 -0.0 -0.0 0.0; -0.0 -0.0 -0.0 0.0; 0.0 0.0 -0.0 m2*g; 0.0 -0.0 0.0 0.0; 0.0 -0.0 0.0 0.0; 0.0 0.0 -0.0 0.0;
0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; -0.0 -0.0 -0.0 -0.0; -0.0 -0.0 0.0 m2*s2; -0.0 -0.0 -0.0 0.0; 0.0 0.0 -0.0 -0.0; 0.0 0.0 0.0 -0.0; -0.0 -0.0 0.0 0.0; -0.0 -0.0 -0.0 -0.0; 0.0 0.0 0.0 -0.0; -0.0 -0.0 -0.0 -0.0; 0.0 0.0 -m2 -0.0; -0.0 0.0 -0.0 -0.0; -0.0 -0.0 0.0 -0.0; 0.0 -0.0 0.0 -0.0]


if !@isdefined(basis)
    h = verkettung(elementarfuntionen, verkettungsarten, 4, true)
    @time basis = Basis(h, u)
end
println("\n Sindy naive nominal")
p_39_ident = sindy_naive(X, DX_, basis, 0.2)
calc_param_ident_error(p_39_nom, p_39_ident)

simultaion = false
if simultaion
    X_, _ , _ = create_data(wp, u0)

    z1 = SIR(p_39_nom, basis)
    @derivatives D'~t
    eqs1 = [D(u[1]) ~ z1[1],
           D(u[2]) ~ z1[2],
           D(u[3]) ~ z1[3],
           D(u[4]) ~ z1[4]]
    de1 = ODESystem(eqs1,t,u,[])
    fu1 = ODEFunction(de1)
    function approx_nom(du, u, p, t)
        z = fu1(u, [], t)
        du[1] = z[1]
        du[2] = z[2]
        du[3] = z[3]
        du[4] = z[4]
    end
    a_prob_nom = ODEProblem(approx_nom, u0, tspan, [])
    sol_nom = solve(a_prob_nom, DP5(),abstol=1e-9, reltol=1e-9, saveat = dt)

    z2 = SIR(p_39_ident, basis)
    @derivatives D'~t
    eqs2 = [D(u[1]) ~ z2[1],
           D(u[2]) ~ z2[2],
           D(u[3]) ~ z2[3],
           D(u[4]) ~ z2[4]]
    de2 = ODESystem(eqs2,t,u,[])
    fu2 = ODEFunction(de2)
    function approx_ident(du, u, p, t)
        z = fu2(u, [], t)
        du[1] = z[1]
        du[2] = z[2]
        du[3] = z[3]
        du[4] = z[4]
    end
    a_prob_ident = ODEProblem(approx_ident, u0, tspan, [])
    sol_ident = solve(a_prob_ident, DP5(), abstol=1e-9, reltol=1e-9, saveat = dt)

    display(plot(X_', label = "original"))
    display(plot!(sol_nom', label = "nom"))
    display(plot!(sol_ident', label = "ident"))

    display(plot(X_' - sol_nom', title = "error =0?"))
    display(plot(X_' - sol_ident', title = "error ident"))
    display(plot(sol_nom' - sol_ident', title = "error 3"))
end


function test(t)
    return -101.93374f0 .* sin.(t) .* inv.(3.34f0 .+ 0.8512f0 .* sin.(t) .^ 2) .+
    -50.966877f0 .* sin.(t) .^ 3 .* inv.(3.34f0 .+ 0.8512f0 .* sin.(t) .^ 2) .+
    -50.966866f0 .* sin.(t) .* cos.(t) .^ 2 .* inv.(3.34f0 .+ 0.8512f0 .* sin.(t) .^ 2)
end
function soll(t)
    return -g*(m1+m2)/s2 .*sin.(t) .*inv.(3.34f0 .+ 0.8512f0 .* sin.(t) .^ 2)
end
#
#
# r = rank(basis(X)')
# sing = svd(basis(X)')
# U = copy(sing.U)
# S = copy(sing.S)
# Vt = copy(sing.Vt)
# for i ∈ 1:length(S)
#     S[i] *= S[i] < 0.1 ? 0 : 1
# end
#
# Sd = Diagonal(ones(eltype(S), length(S))).*S
# Ad = Diagonal(Diagonal(ones(eltype(S), length(S))).*u[1])
# P = (U*Sd)[:,1:28]*Vt[1:28,:]
# a = 0
# for i ∈ 1:r
#      if norm(P[:,i]-basis(X)'[:,i],2) > 0.1
#          println(i)
#      end
# end
# is = []
# Θ = copy(basis(X)')
# P=[]
# function select_lin_dep(Θ)
#     for j ∈ 1:(size(Θ)[2]-r)
#         P = []
#         for i ∈ 1:size(Θ)[2]
#             P = cat(Θ[:,1:i-1], Θ[:,i+1:end], dims = 2)
#             println(size(P))
#             if rank(P) == r # diese Spalten sind linear abhängig
#                 push!(is, j-1+i)
#                 println(i)
#                 break
#             end
#         end
#         Θ = copy(P)
#     end
# end
# select_lin_dep(Θ)


h = verkettung(elementarfuntionen, verkettungsarten, 4, true)
Θ = copy(basis(X)')
is = []
rang_abfall = 0
function sel_lin_indep(Θ, rang_abfall, h)
    is = []
    for i ∈ 2:size(Θ)[2]
        r_ = rank(Θ[:, 1:i])
        if r_ < i - rang_abfall
            push!(is, i)
            rang_abfall += 1
        end
    end
    h_ = copy(h)
    deleteat!(h_, is)
    return h_
end
b = Basis(sel_lin_indep(Θ, 0, h), u)
p_28_ident = sindy_naive(X, DX_, b, 0.2)
calc_param_ident_error(p_28_nom, p_28_ident)

p_28_ident_zen = sindy_naive(X, x_dot, b, 0.2)
calc_param_ident_error(p_28_nom, p_28_ident_zen)

for i ∈ 1:size(Θ)[2]
    println(i, " r= ", rank(Θ[:,1:i]))
end

# for i ∈ 1:size(Θ)[2]
#     P = cat(Θ[:,1:i-1], Θ[:,i+1:end], dims = 2)
#     if rank(P) == r # diese Spalten sind linear abhängig
#         push!(is, i)
#         println(i)
#     end
# end

# for k ∈ 2:2
#     c = collect(combinations(1:length(basis.basis), k))
#     for i ∈ 1:length(c)
#         M = nothing
#         for j ∈ 1:length(c[i])
#             if M == nothing
#                 M = basis(X)'[:,j]
#             else
#                 M = cat(M, basis(X)'[:,j], dims=2)
#             end
#         end
#         # print(size(M))
#         if rank(M)<k
#             println(k," ", i)
#         end
#     end
# end

# Diagonal(Ad)










# x = Sym("x")
# SymPy.subs
# f(x) = sin(x)
# u₁= Sym("u₁")
# u₃= Sym("u₃")
# f3_ = lambdify(f3)
# primitive Library ########################################################
# bb = [sin(u[1]), sin(u[1])*cos(u[1]), u[3]^2*sin(u[1]), u[3]^2*sin(u[1])*cos(u[1]) ]
#
# temp = Operation[]
# comb = collect(combinations(bb,1))
# for i ∈ 1:length(comb)
#     push!(temp, frac(comb[i], m1+m2*sin(u[1])^2)...)
# end
# push!(temp, u[3:4]...)
# p_prim_nom = Float32[-0.0 -0.0 -g/s2*(m1+m2) 0.0; 0.0 0.0 -0.0 g*m2; 0.0 0.0 0.0 m2*s2; -0.0 -0.0 -m2 0.0; 1.0 0.0 -0.0 0.0; 0.0 1.0 -0.0 -0.0]
# b = Basis(temp, u)
#
# println("\n Sindy naive nominal primitiv")
# @time Ψ_prim = sindy_naive(X, DX_, b, 0.2)
# @time print_equations(Ψ_prim, show_parameter = true)
# p_prim_ident = Ψ_prim.coeff
# param = parameters(Ψ_prim)
#
#
#
# # Reibungsidentifiktaion #####################################################
# ff = [tanh(u[3]), u[3], tanh(u[3])*cos(u[1]), u[3]*cos(u[1])] #sin(u[1]), sin(u[1])*cos(u[1]), u[3]^2*sin(u[1]), u[3]^2*sin(u[1])*cos(u[1]),
#
# temp = Operation[1]
# comb = collect(combinations(ff,1))
# for i ∈ 1:length(comb)
#     push!(temp, frac(comb[i], m1+m2*sin(u[1])^2)...)
# end
# push!(temp, u[3:4]...)
# bf = Basis(temp, u)
#
# println("\n Sindy naive nominal primitiv")
# @time Ψ_teil = sindy_naive(Xf, DX_f .- DX_, bf, 0.2)
# @time print_equations(Ψ_teil, show_parameter = true)
# p_teil_ident = Ψ_teil.coeff
# param_teil = parameters(Ψ_teil)



#
#
# Create function
# unknown_sys = ODESystem(Ψ_prim)
# unknown_eq = ODEFunction(unknown_sys)
# ODESystem()
# ODEFunction()
# Build a ODE for the estimated system
