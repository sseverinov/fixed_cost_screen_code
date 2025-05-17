using SummationByPartsOperators
using NLsolve
using LinearAlgebra
using CairoMakie

struct QTauProblem{F, I}
    dtm::Matrix{F}
    dtn::Matrix{F}
    gridm::Vector{F}
    gridn::Vector{F}
    m::I
    n::I
    renormalization_factor::F
    C::F
end

function boundaryproblem(m, n, renormalization_factor, C)
    F = promote_type(typeof(renormalization_factor), typeof(C))

    dtm = Matrix(legendre_derivative_operator(zero(F), one(F), m))
    dtn = Matrix(legendre_derivative_operator(zero(F), one(F), n))

    gridm = SummationByPartsOperators.grid(legendre_derivative_operator(zero(F), one(F), m))
    gridn = SummationByPartsOperators.grid(legendre_derivative_operator(zero(F), one(F), n))

    return QTauProblem(dtm, dtn, gridm, gridn, m, n, renormalization_factor, C)
end

"Function to apply the derivative operator"
function apply_derivative_operator!(eq, dt, u, scale)
    mul!(eq, dt, u, scale, 0)
end

# {at → -1 + 2 tt / tm, bt → -1 + 2 tm}
function toab(tt, tm)
    return atanh.((2tt / tm - 1, 2tm - 1)) # at, bt
end

# {tt → 1/4 + at / 4 + bt / 4 + at * bt / 4, tm → 1 / 2 + bt / 2}
function tothetas(a, b)
    at = tanh(a)
    bt = tanh(b)
    return 1 / 4 + at / 4 + bt / 4 + at * bt / 4, 1 / 2 + bt / 2
end

"Function to compute equations for the first interval"
function compute_first_interval_equations(eq, u, (a, b), params::QTauProblem)
    m, dtm, gridm = params.m, params.dtm, params.gridm

    theta_til, theta_m = tothetas(a, b)
    theta = (linear_interpolate(x, theta_til, theta_m) for x in gridm)

    diff_factor = 4 / ((1 - tanh(a)) * (1 + tanh(b)))
    @views begin
        apply_derivative_operator!(eq[1:m, :], dtm, u[1:m, :], diff_factor)
        
        tau1 = u[1:m, 1]
        Q1 = u[1:m, 2]
        
        @. eq[1:m, 1] -= (theta - tau1) / (tau1 - Q1)
        @. eq[1:m, 2] -= Q1 / (tau1 - Q1)
        
        @. eq[2:m+1, 3] = u[1:m, 3] - u[2:m+1, 3]
        @. eq[2:m+1, 4] = u[1:m, 4] - u[2:m+1, 4]
    end
end

"Function to compute equations for the second interval"
function compute_second_interval_equations(eq, u, (a, b), params::QTauProblem)
    m, n, gridn = params.m, params.n, params.gridn 

    _, theta_m = tothetas(a, b)
    theta = (linear_interpolate(x, theta_m, 1) for x in gridn)

    diff_factor = 2 / (1 - tanh(b))
    @views begin
        apply_derivative_operator!(eq[m+1:n+m, :], params.dtn, u[m+1:n+m, :], diff_factor)
        
        tau1 = u[m+1:n+m, 1]
        Q1 = u[m+1:n+m, 2]
        tau2 = u[m+1:n+m, 3]
        Q2 = u[m+1:n+m, 4]
        
        @. eq[m+1:n+m, 1] -= (theta - tau1) / (tau1 - Q1)
        @. eq[m+1:n+m, 2] -= (Q1 - Q2) / (tau1 - Q1)
        
        @. eq[m+1:n+m, 3] -= (tau1 - tau2) / (tau2 - Q2) * (theta - Q1) / (tau1 - Q1)
        @. eq[m+1:n+m, 4] -= Q2 / (tau2 - Q2) * (theta - Q1) / (tau1 - Q1)
    end
end

"Function to compute continuous field equations"
function compute_continuous_field_equations(eq, u, params::QTauProblem)
    m = params.m
    @. eq[m+1, :] = u[m+1, :] - u[m, :]
end

"Main equation function"
function ode_residual!(eq, u, (a, b), params::QTauProblem)
    compute_first_interval_equations(eq, u, (a, b), params)
    compute_second_interval_equations(eq, u, (a, b), params)
    compute_continuous_field_equations(eq, u, params)
    return eq
end

"Function to compute the residual"
function compute_residual(y, x, params::QTauProblem)
    m, n, renormalization_factor, C = params.m, params.n, params.renormalization_factor, params.C
    a, b = x[4*(m+n)+1], x[4*(m+n)+2]

    @views begin
        qts_u = reshape(x[1:4*(m+n)], m+n, 4)
        qts_eq = reshape(y[1:4*(m+n)], m+n, 4)
    end
    ode_residual!(qts_eq, qts_u, (a, b), params)

    theta_til = (1 + tanh(a)) * (1 + tanh(b)) / 4
    qts_eq[1, 1] = qts_u[n+m, 1] - (theta_til + renormalization_factor)
    qts_eq[1, 2] = qts_u[1, 1] - qts_u[n+m, 3]
    qts_eq[1, 3] = qts_u[n+m, 2] - (theta_til - renormalization_factor)
    qts_eq[1, 4] = qts_u[1, 2] - qts_u[n+m, 4]

    y[4*(m+n)+1] = qts_u[m+1, 4] - qts_u[m+1, 3] - 2 * renormalization_factor
    y[4*(m+n)+2] = qts_u[m+1, 3] * (qts_u[m+1, 1] - qts_u[m+1, 3]) - C
    return y
end

"Constant initial guesses"
function create_initial_constant_guess(inivals, m, n, theta_til0, theta_m0)
    u0 = mapreduce(vcat, inivals) do inival
        fill(inival, n + m)
    end

    push!(u0, toab(theta_til0, theta_m0)...)
    return u0
end

"Function to create the initial guess"
function create_initial_profile_guess((tau1f, Q1f, tau2f, Q2f), m, n, theta_til0, theta_m0)
    thetas = tauq_grid(m, n, theta_til0, theta_m0)

    u0 = vcat(
              tau1f.(thetas),
              Q1f.(thetas),
              fill(tau2f(thetas[m+1]), m),
              tau2f.(thetas[end-n+1:end]),
              fill(Q2f(thetas[m+1]), m),
              Q2f.(thetas[end-n+1:end]),
             )

    push!(u0, toab(theta_til0, theta_m0)...)
    return u0
end

"linear interpolate between s0 and s1 where x=0 -> s1 and x=1->s1"
linear_interpolate(x, s0, s1) = s0 * (1 - x) + s1 * x # x is between 0 to 1

function tauq_grid(m, n, theta_til, theta_m)
    F = promote_type(typeof(theta_til), typeof(theta_m))
    opm = legendre_derivative_operator(zero(F), one(F), m)
    opn = legendre_derivative_operator(zero(F), one(F), n)

    return [
        linear_interpolate.(SummationByPartsOperators.grid(opm), theta_til, theta_m)
        linear_interpolate.(SummationByPartsOperators.grid(opn), theta_m, 1)
    ]
end

function grid(params::QTauProblem{F}, theta_til, theta_m) where F
    m, n = params.m, params.n
    return tauq_grid(m, n, theta_til, theta_m)
end

struct QTauSolution{R, P, F}
    result::R
    params::P
    t2::Vector{F}
    t12::Vector{F}
    tau1::Vector{F}
    Q1::Vector{F}
    tau2::Vector{F}
    Q2::Vector{F}
    theta_til::F
    theta_m::F
end

function QTauSolution(result::NLsolve.SolverResults, params)
    a = result.zero[end-1]
    b = result.zero[end]
    tilde_theta, tilde_m = tothetas(a, b)

    t = grid(params, tilde_theta, tilde_m)
    m, n = params.m, params.n

    t2 = t[m+1:end]
    t12 = [t[1:m]; t[m+2:end]]

    qtmat = reshape(result.zero[1:4*(m+n)], :, 4)
    tau1 = [qtmat[1:m, 1]; qtmat[m+2:end, 1]]
    Q1 = [qtmat[1:m, 2]; qtmat[m+2:end, 2]]
    tau2 = qtmat[m+1:end, 3]
    Q2 = qtmat[m+1:end, 4]

    return QTauSolution(
        result, params,
        t2, t12,
        tau1, Q1,
        tau2, Q2,
        tilde_theta, tilde_m,
    )
end

function solve(params::QTauProblem, initial_guess; nlsolve_kwargs = (xtol = 1e-3,))
    m, n = params.m, params.n
    function residuals!(F, x)
        compute_residual(F, x, params)
    end

    result = nlsolve(residuals!, initial_guess; nlsolve_kwargs...)

    if !converged(result)
        @warn "Q Tau Problem did not converge" 
    end

    return QTauSolution(result, params)
end

function plot_solution(sol::QTauSolution)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"t", ylabel=L"$Q$ or $\tau$")

    lines!(ax, sol.t12, sol.tau1, label=L"\tau_1")
    lines!(ax, sol.t12, sol.Q1, label=L"Q_1")
    lines!(ax, sol.t2, sol.tau2, label=L"\tau_2")
    lines!(ax, sol.t2, sol.Q2, label=L"Q_2")

    vlines!(ax, sol.theta_til, linestyle=:dash, label=L"\tilde\theta")
    vlines!(ax, sol.theta_m, linestyle=:dot, label=L"\theta^M")
    
    axislegend(ax, position = :rb)

    ylims!(ax, low = -0.1)
    xlims!(ax, low = max(0, sol.theta_til) - 0.1, high = 1.1)

    return fig
end

# Example Calls
# example_solve_and_plot_qtau(50, 50, 0.5, 0.75, 0.001, 0.06, [0.7044281876320305, 0.6724368586836897, 0.19430625167743398, 0.6233202229504847])

# example_solve_and_plot_qtau(30, 30, 0.1, 0.9, 0.01, 0.06, [0.8601022352115433, 0.5373859066843202, 0.6853245544826759, 0.8914789961319243])

# example_solve_and_plot_qtau(50, 50, 0.1, 0.8, 0.001, 0.07, [0.16580919251801107, 0.1360974511768921, 0.20510279164620715, 0.12738425664551534])

# example_solve_and_plot_qtau(60, 60, 0.1, 0.9, 0.001, 0.07, [0.8, 0.7, 0.4, 0.3])

function example_solve_and_plot_qtau(m, n, tt, tm, renormalization_factor, C, ini)
    params = boundaryproblem(m, n, renormalization_factor, C)

    # Choose an initial guess strategy (constant or profile)
    initial_guess = create_initial_constant_guess(ini, m, n, tt, tm) # Example constant guess
    # initial_guess = create_initial_profile_guess([x -> x, x -> x^2, x -> x^3, x -> x^4], n, m, 0.5, 0.6) # Example profile guess

    solution = solve(params, initial_guess)

    # Plotting
    fig = plot_solution(solution)
    display(fig)

    return solution
end

function run_qtausolver(m, n, tt, tm, renormalization_factor, C, tauq_ini_values)
    params = boundaryproblem(m, n,  tt, tm, renormalization_factor, C)
    initial_guess = create_initial_constant_guess(tauq_ini_values, n, m, tt, tm)
    return solve(params, initial_guess)
end
