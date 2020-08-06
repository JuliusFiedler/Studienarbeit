# Fall 1
a = [1 2]
a = cat(a, a, dims=2)
a = cat(a, a, dims=2)
println(a)

# Fall 2
for i ∈ (1:2)
    a = cat(a, a, dims=2)
end
println(a)

# Fall 3
let a = [1 2]
    for i ∈ (1:2)
        a = cat(a, a, dims=2)
    end
    println(a)
end
printn(a)

function mul_tra!(X, DX_, x_dot, u0)
    min_u, = findmin(u0)
    max_u, = findmax(u0)
    len = length(u0)
    no_tr = 2
    A = similar(X)
    A .= X
    for i ∈ (1:no_tr)
        Random.seed!(i)
        u0_ = rand(Float32, len) .*(max_u - min_u) .+ min_u
        X_, DX__, x_dot_ = create_data(u0_)
        A_ = similar(X_)
        A_ .= X_
        A = cat(A, A_, dims = 2)
        DX_ = cat(DX_, DX__, dims = 2)
        x_dot = cat(x_dot, x_dot_, dims = 2)
    end
    X = A
    return X, DX_, x_dot
end
