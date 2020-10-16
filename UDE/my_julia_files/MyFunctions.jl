module MyFunctions

@time using OrdinaryDiffEq
@time using ModelingToolkit
@time using DataDrivenDiffEq
@time using LinearAlgebra
@time using DiffEqSensitivity, Optim
@time using DiffEqFlux, Flux
@time using Plots
@time using Combinatorics
@time using Random

export calc_centered_difference
export SIR
export sindy_naive, sindy_mpl
export calc_param_ident_error, calc_param_ident_error_div_by_ev
export wp, wp_fric, lotka, lorenz, roessler, wp_lin
export create_data
export mul_tra!


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
    return string(rel_error, msg)
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

function create_data(sys, u0, tspan, p_, dt = 0.1)
    prob = ODEProblem(sys, u0, tspan, p_)
    X = solve(prob, Vern7(), abstol=1e-9, reltol=1e-9, saveat = dt)
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
    x_dot = calc_centered_difference(X, dt)
    return X, DX_, x_dot
end

function mul_tra!(sys, X, DX_, x_dot, u0, tspan, p_, dt = 0.1, no_tr = 10)
    X, DX_, x_dot = create_data(sys, u0, tspan, p_, dt)
    min_u, _ = findmin(u0)
    max_u, _ = findmax(u0)
    len = length(u0)
    A = similar(X)
    A .= X

    for i ∈ (1:(no_tr-1))
        Random.seed!(i)
        u0_ = rand(Float32, len) .*(max_u - min_u) .+ min_u
        println(u0_)
        X_, DX__, x_dot_ = create_data(sys, u0_, tspan, p_, dt)
        A_ = similar(X_)
        A_ .= X_
        A = cat(A, A_, dims = 2)
        DX_ = cat(DX_, DX__, dims = 2)
        x_dot = cat(x_dot, x_dot_, dims = 2)
    end
    X = A
    return X, DX_, x_dot
end



end # Module
