@time using OrdinaryDiffEq
@time using ModelingToolkit
@time using DataDrivenDiffEq
@time using LinearAlgebra, DiffEqSensitivity, Optim
@time using DiffEqFlux, Flux
@time using Plots
@time using Combinatorics
using Random
gr()
println("-------------------------------------------------------------------")

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

function sindy_naive(X, Ẋ, basis, λ = th)
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
    @time sir = SparseIdentificationResult(Ξ, basis, 1, SR3(), true, Ẋ', X)
    return sir
end

function calc_param_ident_error(p_n, p_i)
    digits = 5
    s = 0
    t = 0
    msg = ""
    for a ∈ (1:length(p_n))
        if p_n[a] != 0
            s += ((p_n[a] - p_i[a]) / p_n[a])^2
        elseif p_i[a] != 0
            s += ((p_n[a] - p_i[a]) / p_i[a])^2
            msg = "*"
        end
        t += (p_n[a] - p_i[a])^2
    end
    rel_error = round(sqrt(s/length(p_n)), digits=digits)
    abs_error = round(sqrt(t/length(p_n)), digits=digits)
    println("   RMS absoluter Fehler: ", abs_error)
    println("   RMS relativer Fehler: ", rel_error, msg)
    return [abs_error, string(rel_error, msg)]
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

maxiter = 30
p_ = Float32[3.34, 0.8512, 9.81, 0.26890449]
tspan = (0.0f0,5.0f0)
u0 = Float32[-3, 3, 0.5 ,0.5]
prob = ODEProblem(wp, u0,tspan, p_)
dt = 0.1
m1, m2, g, s2 = p_
multiple_trajectories = true

function create_data(sys, u0)
    prob = ODEProblem(sys, u0, tspan, p_)
    X = solve(prob, DP5(), abstol=1e-12, reltol=1e-12, saveat = dt)#für wp dp5, vorher stand hier Vern7
    if sys == wp
        DX_ =  [ X[3,:]';
                X[4,:]';
                -(g .*m1 + g .*m2 .+ m2.*X[3,:]'.^2 .*s2 .*cos.(X[1,:]')).*sin.(X[1,:]') ./(s2 .*(m1 .+ m2 .*sin.(X[1,:]') .^2));
                (m2 .*(g .*cos.(X[1,:]') .+ X[3,:]' .^2 .*s2) .*sin.(X[1,:]')) ./(m1 .+ m2 .*sin.(X[1,:]') .^2)]
    elseif sys == wp_fric
        DX_ =  [ X[3,:]';
                X[4,:]';
                -((g .*m1 + g .*m2 .+ m2.*X[3,:]'.^2 .*s2 .*cos.(X[1,:]')).*sin.(X[1,:]') .+(m1+m2).*(R(X[3,:]))' /(m2.*s2))./(s2 .*(m1 .+ m2 .*sin.(X[1,:]') .^2));
                (m2 .*(g .*cos.(X[1,:]') .+ X[3,:]' .^2 .*s2) .*sin.(X[1,:]') .+ (R(X[3,:]))'.*cos.(X[1,:]')./s2) ./(m1 .+ m2 .*sin.(X[1,:]') .^2)]
    end
    x_dot = calc_centered_difference(X, dt)
    return X, DX_, x_dot
end

function mul_tra!(sys, X, DX_, x_dot, u0)
    X, DX_, x_dot = create_data(sys, u0)
    min_u, = findmin(u0)
    max_u, = findmax(u0)
    len = length(u0)
    no_tr = 10
    A = similar(X)
    A .= X

    for i ∈ (1:(no_tr-1))
        Random.seed!(i)
        u0_ = rand(Float32, len) .*(max_u - min_u) .+ min_u
        X_, DX__, x_dot_ = create_data(sys, u0_)
        A_ = similar(X_)
        A_ .= X_
        A = cat(A, A_, dims = 2)
        DX_ = cat(DX_, DX__, dims = 2)
        x_dot = cat(x_dot, x_dot_, dims = 2)
    end
    X = A
    return X, DX_, x_dot
end

X, DX_, x_dot = create_data(wp, u0)
Xf, DX_f, x_dotf = create_data(wp_fric, u0)

if multiple_trajectories
    X, DX_, x_dot = mul_tra!(wp, X, DX_, x_dot, u0)
    Xf, DX_f, x_dotf = mul_tra!(wp_fric, X, DX_, x_dot, u0)
end


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
    for i ∈ (2:depth)
        comb = collect(with_replacement_combinations(h,i))
        for j ∈ (1:length(comb))
            for k ∈ 1:length(verk)
                push!(z, verk[k](comb[j]))
            end
        end
    end
    push!(z, h...)
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
        push!(temp, u...)
    end
    # return h
    return temp
end

# größe Library ############################################################
@time h = verkettung(elementarfuntionen, verkettungsarten, 4, true)
@time basis = Basis(h, u)
println("\n Sindy naive nominal")
@time Ψ = sindy_naive(X, DX_, basis, 0.2)
@time print_equations(Ψ, show_parameter = true)
p_ident = Ψ.coeff

# primitive Library ########################################################
bb = [sin(u[1]), sin(u[1])*cos(u[1]), u[3]^2*sin(u[1]), u[3]^2*sin(u[1])*cos(u[1]) ]

temp = Operation[]
comb = collect(combinations(bb,1))
for i ∈ 1:length(comb)
    push!(temp, frac(comb[i], m1+m2*sin(u[1])^2)...)
end
push!(temp, u[3:4]...)
p_prim_nom = Float32[-0.0 -0.0 -g/s2*(m1+m2) 0.0; 0.0 0.0 -0.0 g*m2; 0.0 0.0 0.0 m2*s2; -0.0 -0.0 -m2 0.0; 1.0 0.0 -0.0 0.0; 0.0 1.0 -0.0 -0.0]
b = Basis(temp, u)

println("\n Sindy naive nominal primitiv")
@time Ψ_prim = sindy_naive(X, DX_, b, 0.2)
@time print_equations(Ψ_prim, show_parameter = true)
p_prim_ident = Ψ_prim.coeff
param = parameters(Ψ_prim)



# Reibungsidentifiktaion #####################################################
ff = [tanh(u[3]), u[3], tanh(u[3])*cos(u[1]), u[3]*cos(u[1])] #sin(u[1]), sin(u[1])*cos(u[1]), u[3]^2*sin(u[1]), u[3]^2*sin(u[1])*cos(u[1]),

temp = Operation[1]
comb = collect(combinations(ff,1))
for i ∈ 1:length(comb)
    push!(temp, frac(comb[i], m1+m2*sin(u[1])^2)...)
end
push!(temp, u[3:4]...)
bf = Basis(temp, u)

println("\n Sindy naive nominal primitiv")
@time Ψ_teil = sindy_naive(Xf, DX_f .- DX_, bf, 0.2)
@time print_equations(Ψ_teil, show_parameter = true)
p_teil_ident = Ψ_teil.coeff
param_teil = parameters(Ψ_teil)



#
#
# Create function
unknown_sys = ODESystem(Ψ_prim)
unknown_eq = ODEFunction(unknown_sys)

# Build a ODE for the estimated system
function approx(du, u, p, t)
    z = unknown_eq(u, p, t)
    du[1] = z[1]
    du[2] = z[2]
    du[3] = z[3]
    du[4] = z[4]
end

X_, _ , _ = create_data(u0)
a_prob = ODEProblem(approx, u0, tspan, param)
a_solution = solve(a_prob, DP5(), saveat = dt)
display(plot(X_', label = "original"))
display(plot!(a_solution', label = "ident"))

calc_param_ident_error(p_prim_nom, p_prim_ident)
