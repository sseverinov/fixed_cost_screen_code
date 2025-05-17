### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 28b75488-9997-42e0-b2d5-e69a75960fce
using SummationByPartsOperators,NLsolve, CairoMakie, ColorSchemes

# ╔═╡ 4671adce-b2f3-4834-b1d2-d9769fdea976
using Accessors, DataFrames, Latexify

# ╔═╡ 4390fb37-e5de-47d4-a61a-3ee01e833a9c
using PlutoUI

# ╔═╡ 5f09efcb-a077-436e-8e93-b78d0b47581d
import Interpolations

# ╔═╡ 6b1c78f2-a53d-40b7-b4db-57d4c4798e11
TableOfContents(title = "M=1 and M=2 Plots")

# ╔═╡ 31fbf75d-44cd-4ce2-a38d-5ccc8b6b3973
md"""
Copy and pasted the `include("../qtau_solver.jl")` code for better portability.
The cell is hidden because it is large. **Note** any changes to the qtau_solver.jl file will not be reflected in this notebook.
"""

# ╔═╡ 06602b0a-34ff-499a-8c36-7d575b49fb3c
begin
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

end

# ╔═╡ 722ec8c2-7a95-4616-8385-3e2c7f29c47a
md"""
## Global Plotting Parameters
"""

# ╔═╡ 8027abff-2405-419b-a3c8-77f7b8a63c1a
let mycolors = Makie.wong_colors()
	permute!(mycolors, [3,2,1,4,5,6,7])
	set_theme!(
	size=(192*6, 108*6),
	 palette = (
        color = mycolors, #[:red, :green, :blue, :orange, :purple],  # Your custom array of colors
    ),
	fontsize = 24,
);
end

# ╔═╡ bb6a3d85-137c-43c9-b1df-bc777274dcda
plotting_ngrid_points = 100

# ╔═╡ 70ac10e0-7272-4f27-87fc-035b387e51bb
m2minCCs = range(0.04, 0.09, length=10)

# ╔═╡ f7b6b26c-ba00-4cd7-998c-c3f802d3a9aa
"""
The black and white theme
"""
bw_theme = merge(Theme(
    palette = (
        color = repeat([:black], 20), # Ensures all lines are black
        marker = (:circle, :rect, :utriangle, :star5, :xcross, :diamond, :pentagon, :hexagon),
    ),
    Axis = (
        spinecolor = :black,
        tickcolor = :black,
        gridcolor = (:gray90, 0.75),
        xlabelcolor = :black,
        ylabelcolor = :black,
    ),
    Legend = (framecolor = :black,)
), Makie.current_default_theme())

# ╔═╡ 30b77702-548c-4ec7-9b8c-5d3f8a979f5f
md"""
## Model Parameters
"""

# ╔═╡ 40779d47-cddc-49a6-9a11-46f494853072
model_parameters = [
	(C = 0.04, M = 2),
	(C = 0.065, M = 2),
	# (C = 0.09, M = 2),
	(C = 0.09, M = 1),
	(C = 0.15, M = 1),
	(C = 0.22, M = 1),
	]

# ╔═╡ cb1de805-72d8-4a44-9027-2f15e5115966
md"""
## Solver Parameters
"""

# ╔═╡ 9dfeb033-96e9-46e9-99a5-26cfa556242e
md"""
Number of grid points.

- `m`: number of gridpoints for $[\tilde\theta, \theta^M]$
- `n`: number of gridpoints for $[\theta^M, 1]$
- `theta_m0`, `theta_tilde0`: Initial thetas
- `renormalization_factor`: a factor that keeps the Raphson-Newton method stable
- `xtol` relative tolerance for the solver.
"""

# ╔═╡ 3d4f6b18-a282-4462-a715-63bc933dce15
n = 55

# ╔═╡ 10732e49-b440-442f-9bf0-b588704d59c9
m = 55

# ╔═╡ c88775ab-2c55-427c-b8b8-8c0f0bd5cc78
theta_tilde0 = 0.75

# ╔═╡ fa64da8f-f3f6-4122-a5dc-044f4057f0d1
theta_m0 = 0.95

# ╔═╡ 975071ab-719f-4bd9-a57a-14a6c4d3d648
renormalization_factor = 0.003

# ╔═╡ 758b14d5-71b0-4bf0-acf0-5dc525346488
xtol = 1e-5

# ╔═╡ a2952c8c-425b-4e05-a379-8dbfb20b7708
md"""
# Plotting
"""

# ╔═╡ 63df4212-f263-4fcb-9438-aa5a1f2d4afd
md"""
**Warning:** Smoothing function, `smoooth`, is used here to compensate for the `renormalization_factor`
"""

# ╔═╡ ec560f14-a8d3-4400-ae39-eae91cb79401
md"""
## A plot of five pairs of functions $\tau$ and $q$
"""

# ╔═╡ e24dd6d9-55ca-47a6-a8e4-746d54a794d2
md"""
## A plot of  five $\tau$ functions
"""

# ╔═╡ 19e4055f-b0ce-4fc7-a87d-9ec635af5d18
md"""
## A plot of five q functions
"""

# ╔═╡ 758e9da4-4c63-460f-83ea-75b19cc9bc4e
md"""
## Task 1: Finding the minimum $C$ (of $M = 2$)
"""

# ╔═╡ 07785a81-04e2-4467-a94a-e72d7ba4ed90
md"""
## Task 2: "Plotting $\theta$ values and $\tau(1)$"
"""

# ╔═╡ 543f96e6-702a-471f-bdaf-ac7fa1e78cac
md"""
### $\theta$ Table
"""

# ╔═╡ 1293036b-438e-44a7-bf98-5a801c2e7980
md"""
## Task 3: "Old Plotting $\tau$ (for $M = 2$)"
"""

# ╔═╡ b186742f-27dc-44fb-845d-a8c0e88f1c31
md"""
## Task 4: "Old $q$ Plots"
"""

# ╔═╡ bb680f62-a993-454e-a9d4-afdce50ce401
md"""
## Task 5: "Old $\tau$-$q$ pair plots."
"""

# ╔═╡ cbfe6d7d-0448-4810-b823-e53cbbe31572
md"""
**Note**: This plot is wrong. The $C$ for M=2 is too small, but I have kept it in since it was from the previous notebook. 
"""

# ╔═╡ 11af02ed-d7ea-47df-875c-f7b3b0577847
md"""
# Finding Solutions
"""

# ╔═╡ e825a420-fb13-40c8-8e67-5c41306529d0
md"""
## Finding A Reference Solution

Using a reference solution, I can set it as a educated guess for other C value problems.
"""

# ╔═╡ a8d7b84a-603b-4877-b3a9-c0d63a098e85
# found with trial and error
tauq_ini_values = [0.8, 0.7, 0.4, 0.3]

# ╔═╡ 92ebc267-984a-415d-b4ab-12fa187de59a
Cini = 0.05

# ╔═╡ a66b8e4a-794f-4978-b973-51eb59869d42
reference_prob = boundaryproblem(m, n, renormalization_factor, Cini)

# ╔═╡ 76594335-40ce-4d19-95c0-1be236ca5aa7
reference_sol = let
	u0 = create_initial_constant_guess(tauq_ini_values, m, n, theta_tilde0, theta_m0)
	solve(reference_prob, u0)
end

# ╔═╡ 2c6db510-9afa-4081-b1b6-64bb07245460
# plot_solution(reference_sol)

# ╔═╡ dd044433-8e5f-4ad2-a0e4-f82cc302c575
md"""
# Finding other C solutions
"""

# ╔═╡ 69d474af-b030-444c-bff2-4841cd5f4ed1
md"""
### M2 solutions for $\tau$ and $q$ plots
"""

# ╔═╡ 03372f88-1c7d-48f5-b73d-a754b4b5bb49
m2params = filter(p -> p.M == 2, model_parameters)

# ╔═╡ 2c7afc7e-d1a2-4fa4-ae0d-400fb1b529ed
m2sols = Dict(map(m2params) do (; C)
	prob = Accessors.@set reference_prob.C = C
	u0 = reference_sol.result.zero
	C => solve(prob, u0)
end)

# ╔═╡ 6129104e-d352-4b98-81fa-8351e32c0cdb
md"""
### Solutions for finding minimum $C$

`m2minCsols` is a set of more solutions for $M=2$.
"""

# ╔═╡ 0f39d1fc-3dff-4dfd-bb69-593e5d6e36c9
m2minCsols = map(m2minCCs) do C
	prob = Accessors.@set reference_prob.C = C
	u0 = reference_sol.result.zero
	solve(prob, u0)
end

# ╔═╡ 9174c5e2-3e33-4773-b791-35da82177784
let
	sols = m2minCsols
	dth = [sol.theta_m - sol.theta_til for sol in m2minCsols]
	C = [sol.params.C for sol in m2minCsols]
	
	takeCn = 4
	m, b = [C[1:takeCn] C[1:takeCn] .^ 0] \ dth[1:takeCn]
	global yinter_min_C = -b/m
	f, ax = scatter(C, dth)
	lines!(ax, [0; C], [C * m + b for C in [0; C]])
	vlines!(ax, [yinter_min_C], color = :red, label = "Min C≈ $(round(yinter_min_C, sigdigits=3))")
	scatter!(ax, yinter_min_C, 0.0, color = :red)
	axislegend(ax, position = :rb)
	
	ax.ylabel = L"\theta^M - \tilde\theta"
	ax.xlabel = L"C"
	ax.title = "Linearly Fitting the First $takeCn C's to Find the Minimum Solution"
	f
end

# ╔═╡ ffa31d71-600c-4a9c-9bb8-812f7e16e7e7
let
	Cs = map(sol -> sol.params.C, m2minCsols)
	theta_ms = map(sol -> sol.theta_m, m2minCsols)
	theta_tils = map(sol -> sol.theta_til, m2minCsols)
	tau11s = map(sol -> sol.tau1[end], m2minCsols)
	
	fig = Figure()
	ax = Axis(fig[1,1])
	ax.ylabel = L"\theta"
	ax.xlabel = L"$C$ or $\tau$"
	scatter!(ax, Cs, theta_ms, label = L"\theta^M")
	scatter!(ax, Cs, theta_tils, label = L"\tilde{\theta}")
	scatter!(ax, Cs, tau11s, label = L"\tau^1(1)", alpha = 0.5)
	hlines!(ax, [0.78], label = L"$\tau^1(1)$\n where\n$C=0.09$", color = :black)
	
	Legend(fig[1, 2], fig.content[1, 1])
	ax.title = L"Values of $\tau(1)$ and $\theta~$'s"
	fig
end

# ╔═╡ e896c5c6-876e-4ae5-9bf2-0f264baa5361
let
	Cs = map(sol -> sol.params.C, m2minCsols)
	theta_ms = map(sol -> sol.theta_m, m2minCsols)
	theta_tils = map(sol -> sol.theta_til, m2minCsols)
	tau1_theta_ms = map(sol -> sol.tau1[sol.params.m], m2minCsols)
	tau2_theta_ms = map(sol -> sol.tau2[1], m2minCsols)	
	df = DataFrame(
		L"C" => Cs, 
		L"\theta^M" => theta_ms,
		L"\tilde\theta" => theta_tils,
		L"\tau_1(\theta^M)" => tau1_theta_ms,
		L"\tau_2(\theta^M)" => tau2_theta_ms,
	)
	println(latextabular(df, fmt = "%f.2"))
end

# ╔═╡ f7262323-4401-4fc1-aeb8-ea4e70f86e98
md"""
# Implementations of $M=2$ case
"""

# ╔═╡ 993ecf93-0d57-41f1-a0cb-a1b4577383f5
"""
    m2taugrid(sol)

Constructs a grid of `tau` values from the solution `sol`. It concatenates
the `tau1` values from index `m+1` to the end with `t12` values.

**Arguments**
- `sol`: A solution object containing `tau1` and `t12` arrays.

**Returns**
- A concatenated array of `tau` values.
"""
function m2taugrid(sol)
    vcat(sol.tau1[m+1:end], sol.t12)
end

# ╔═╡ 82b88c4d-633a-4530-97b1-3103f2f59bc8
"""
    m2tauvals(sol)

Interpolates `tau2` values over `t2` and concatenates them with `tau1` values
from the solution `sol`.

**Arguments**
- `sol`: A solution object containing `t2`, `tau2`, and `tau1` arrays.

**Returns**
- A concatenated array of interpolated `tau` values.
"""
function m2tauvals(sol)
    tau2li = Interpolations.linear_interpolation(sol.t2, sol.tau2)
    vcat(tau2li.(sol.t12[m+1:end]), sol.tau1)
end

# ╔═╡ d10f2a6b-2479-4f71-ad04-6ee787f99184
"""
    m2qvals(sol, n::Integer)

Generates a range of `q` values with `n` grid points from `tau1` and 1.

**Arguments**
- `sol`: A solution object containing `tau2`, `Q2`, and `Q1` arrays.
- `n`: An integer specifying the number of grid points.

**Returns**
- A concatenated array of `q` values.
"""
function m2qvals(sol, n::Integer)
    vcat(
        range(0, sol.tau2[begin], length=n+1)[1:end-1],
        sol.Q2,
        sol.Q1,
        range(sol.tau1[end], 1, length=n+1)[2:end]
    )
end

# ╔═╡ 0e93b2a9-6e81-497c-8248-98c096d3bbe5
"""
    m2qgrid(sol, n::Integer)

Constructs a grid of `q` domain elements with `n` grid points from `tau1` and 1.

**Arguments**
- `sol`: A solution object containing `tau2` and `tau1` arrays.
- `n`: An integer specifying the number of grid points.

**Returns**
- A concatenated array of `q` grid points.
"""
function m2qgrid(sol, n::Integer)
    vcat(
        range(0, sol.tau2[begin], length=n+1)[1:end-1],
        sol.tau2,
        sol.tau1,
        range(sol.tau1[end], 1, length=n+1)[2:end]
    )
end

# ╔═╡ 75646561-783c-4de2-b529-78c362628b19
"Helper function to account for the renormalization factor. It shifts the plot points `xs` such that the xs are unique"
function smooth(xs)
	xs = copy(xs)

	for i in 1:length(xs)-1
		if xs[i+1] - xs[i] < 0 
			xs[i+1:end] .-= xs[i+1] - xs[i] 
		end
	end
	return xs
end

# ╔═╡ d53543cd-069a-4295-bd64-74cb09e7dd46
let
	fig = Figure()
	ax = Axis(fig[1, 1])
	for (; C) in m2params
		sol = m2sols[C]
		vlines!(ax, [sol.tau1[sol.params.n]], color=:grey, linestyle=:dash, label=L"\tau_1(\theta)")
		vlines!(ax, [sol.theta_til], color=:grey, linestyle=:dot, label=L"\tilde\theta")
		scatter!(ax, sol.theta_til, sol.tau2[end], color=:grey)

		clabel = L"C = %$(round(sol.params.C, sigdigits=4))"
		lines!(ax, smooth(m2taugrid(sol)), m2tauvals(sol), label=clabel)
	end
	axislegend(ax, position=:rb, merge=true)
	ax.xlabel = L"\theta"
	ax.ylabel = L"\tau"
	fig
end

# ╔═╡ d9e9a00c-d7b5-4067-ab6e-14a43360e9b0
let
	qn = max(m, n)
	fig = Figure()
	ax = Axis(fig[1, 1])
	
	for (; C) in m2params
		sol = m2sols[C]
		clabel = L"C = %$(round(sol.params.C, sigdigits=4))"
		xs = smooth(m2qgrid(sol, qn))
		ys = m2qvals(sol, qn)
		lines!(ax, xs, ys, label=clabel)
	end
	
	axislegend(ax, position=:lc)
	ax.xlabel = L"t"
	ax.ylabel = L"\tau"
	fig
end

# ╔═╡ 0bcb2986-918c-4122-aef7-d50af87369a4
let
	qn = max(m, n)
	fig = Figure()
	ax = Axis(fig[1, 1])
	
	for (; C) in m2params
		sol = m2sols[C]
		clabel = L"C = %$(round(sol.params.C, sigdigits=4))"
		xs = smooth(m2qgrid(sol, qn))
		ys = m2qvals(sol, qn)
		scatter!(ax, xs, ys, label=clabel)
	end
	
	axislegend(ax, position=:lc)
	ax.xlabel = L"t"
	ax.ylabel = L"\tau"
	fig
end

# ╔═╡ 66cc35ca-b110-4cc9-807f-5d31e89004ef
md"""
# Implementations of M=1 case functions.
"""

# ╔═╡ f9fdf572-fea0-4f8a-83dc-26b6794a95e4
import Roots

# ╔═╡ a6c23fbf-abea-4144-9ef4-4d26584d0404
function b_anal(that)
	that == 1 && return -one(float(that))
	-(sqrt(1/5) * that^((sqrt(5) - 1)/2) - sqrt(1/5) * that^(-(sqrt(5) + 1)/2))/(that - ((1 + sqrt(1/5))/ 2) * that^(((sqrt(5) - 1))/2) - ((1 - sqrt(1/5))/ 2) * that^(-(sqrt(5) + 1)/2))
end

# ╔═╡ 1ac0fee2-585f-45db-bce9-f615756f03ab
function F_anal(XC, b, that)
	sqrt5 = sqrt(5)
    term1 = (that^2) / 2
    term2 = sqrt(1/5) * that^((sqrt5 + 1) / 2)
    term3 = sqrt(1/5) * that^(-(sqrt5 - 1) / 2)
    term4 = ((sqrt5 - 1) / (2 * sqrt5)) * that^((sqrt5 + 1) / 2)
    term5 = ((sqrt5 + 1) / (2 * sqrt5)) * that^(-(sqrt5 - 1) / 2)
    result = XC + (b / 2) * (b * (term1 - term2 + term3) + term4 + term5)
    return result
end

# ╔═╡ 303ab0c2-e951-4901-a83e-da81844d3f25
function find_that_anal(
	C, 
	th0lower = 0.01,
	th0higher = 4.0)
	Roots.find_zero(that ->  F_anal(C, b_anal(that), that), (th0lower, th0higher))
end

# ╔═╡ 716e84f8-289e-46a3-bb1e-48e3fe4ca5a0
function theta_anal(t, C, b = b_anal(find_that_anal(C)))
    sqrt5 = sqrt(5)
    sqrt1_5 = sqrt(1/5)
    
    term1 = t
    term2 = ((1 + 3 * sqrt1_5) / 2) * t^((sqrt5 - 1) / 2)
    term3 = ((3 * sqrt1_5 - 1) / 2) * t^(-(sqrt5 + 1) / 2)
    term4 = ((sqrt5 + 1) / (2 * sqrt5)) * t^((sqrt5 - 1) / 2)
    term5 = ((sqrt5 - 1) / (2 * sqrt5)) * t^(-(sqrt5 + 1) / 2)
    
    result = b * (term1 - term2 + term3) + term4 + term5
    return result
end

# ╔═╡ 836a2c06-8cab-4f60-8895-36fc2207bfa2
function Q_anal(t, C, b = b_anal(find_that_anal(C)))
    return -(b/2) * t
end

# ╔═╡ 602329dd-a9e5-4106-a36e-edbf26316a9b
function tau_anal(t, C, b = b_anal(find_that_anal(C)))
    sqrt5 = sqrt(5)
    sqrt1_5 = sqrt(1/5)
    
    term1 = t / 2
    term2 = ((1 + sqrt1_5) / 2) * t^((sqrt5 - 1) / 2)
    term3 = ((1 - sqrt1_5) / 2) * t^(-(sqrt5 + 1) / 2)
    term4 = sqrt1_5 * t^((sqrt5 - 1) / 2)
    term5 = sqrt1_5 * t^(-(sqrt5 + 1) / 2)
    
    result = b * (term1 - term2 - term3) + term4 - term5
    return result
end

# ╔═╡ f7e54217-0bf9-4a2e-8485-46e7347fa05c
let 
	fig = Figure()
	ax = Axis(fig[1,1], yscale=identity)

	Cs = map(p->p.C, filter(p->p.M == 1, model_parameters))
	for (series_ind, C) in enumerate(Cs)
		color = Cycled(series_ind)
		ts = range(find_that_anal(C), 1, length=100)
		lines!(ax, tau_anal.(ts, C), Q_anal.(ts, C), color=color, linewidth=2, label = L"C = %$C") # q
		lines!(ax, theta_anal.(ts, C), tau_anal.(ts, C), color=color, linestyle=:dash, linewidth=2) # τ
	end
	
	lines!(ax, range(0, 1, length=10), range(0, 1, length=10), color=:black)

	axislegend(ax, position=:lt)
	
	ax.xlabel = L"t"
	ax.ylabel = L"$\tau$ or $q$"
	ax.title = L"$\tau$ and $q$ plot for $M=1$ case for several Cs"
	fig
end

# ╔═╡ 8d352d4d-0e7f-4e67-aa54-4a425a7f95db
let
	qn = max(m, n)
	fig = Figure()
	ax = Axis(fig[1, 1])
	

	Cs = map(p->p.C, filter(p->p.M == 1, model_parameters))
	
	for (series_ind, (CM15, sol)) in enumerate(m2sols)
		color = Cycled(series_ind)
		
		ts = range(find_that_anal(CM15), 1, length=100)
		qclabel = L"q_{M=1, C = %$(round(CM15, sigdigits=3))}"
		lines!(ax, tau_anal.(ts, CM15), Q_anal.(ts, CM15), label=qclabel, color = color) # q
		tauclabel = L"\tau_{M=1, C = %$(round(CM15, sigdigits=3))}"
		lines!(ax, theta_anal.(ts, CM15), tau_anal.(ts, CM15), label=tauclabel, linestyle=:dash, color = color) # τ

		color = Cycled(length(m2sols) + series_ind)
		C = sol.params.C
		qclabel = L"q_{M=2, C = %$(round(sol.params.C, sigdigits=3))}"
		lines!(ax, smooth(m2qgrid(sol, qn)), m2qvals(sol, qn), label=qclabel, color=color)
		
		tauclabel = L"\tau_{M=2, C = %$(round(sol.params.C, sigdigits=3))}"
		lines!(ax, smooth(m2taugrid(sol)), m2tauvals(sol), label=qclabel, color=color, linestyle=:dash, linewidth=2)
	end
	Legend(fig[1, 2], fig.content[1, 1])
	ax.xlabel = L"t"
	ax.ylabel = L"$\tau$ or $q$"
	ax.title = L"$\tau$ and $q$ plot with $M=1, 2$ cases"
	fig
end

# ╔═╡ c9c315bd-63fa-4fac-99a2-42f5a5301709
"""
    m2ts(C, n)

Generates a range of values starting from the result of `find_that_anal(C)`
to 1, with a length of 100.

**Arguments**
- `C`: A parameter used in the `find_that_anal` function.
- `n`: An integer, though not used in this function.

**Returns**
- A range of values from `find_that_anal(C)` to 1.
"""
m2ts(C, n) = range(find_that_anal(C), 1, length=100)

# ╔═╡ 3f986b17-fa96-4e2b-a61d-378535ace104
"""
    m1taugrid(C, n)

Computes a grid of `tau` values by applying `theta_anal` to each element
in the range generated by `m2ts(C, n)`.

**Arguments**
- `C`: A parameter used in the `theta_anal` function.
- `n`: An integer, passed to `m2ts`.

**Returns**
- An array of `tau` values computed by `theta_anal`.
"""
m1taugrid(C, n) = theta_anal.(m2ts(C, n), C)

# ╔═╡ 50dddf04-dc51-41d3-97b4-35034990fbb2
"""
    m1tauvals(C, n)

Calculates `tau` values by applying `tau_anal` to each element in the range
generated by `m2ts(C, n)`.

**Arguments**
- `C`: A parameter used in the `tau_anal` function.
- `n`: An integer, passed to `m2ts`.

**Returns**
- An array of `tau` values computed by `tau_anal`.
"""
m1tauvals(C, n) = tau_anal.(m2ts(C, n), C)

# ╔═╡ 29b430c7-e482-4718-93fc-2e72e03f036d
make_taufigure(; linestyles = :solid) = let
	fig = Figure()
	ax = Axis(fig[1, 1])

	ax.xlabel = L"\theta"
	ax.ylabel = L"\tau"

	for (series_ind, (; M, C)) in enumerate(model_parameters)
		@assert M in (1, 2)
		linewidth = M == 1 ? 2.0 : 1.0

		linestyle = if linestyles isa Symbol
			linestyles
		else
			linestyles[series_ind]
		end
		
		taugrid = if M == 1
			m1taugrid(C, plotting_ngrid_points) 
		elseif M == 2
			grd = m2taugrid(m2sols[C])
			smooth(grd)
		end

		tauvals = if M == 1
			m1tauvals(C, plotting_ngrid_points)
		elseif M == 2
			m2tauvals(m2sols[C])
		end

		lines!(ax, taugrid, tauvals, label = L"M=%$M, C=%$C", linewidth=linewidth, linestyle=linestyle)
	end
	
	axislegend(ax, position = :lt, merge = true)
	fig
end

# ╔═╡ 93141f29-0b47-4353-8241-b9ca791448ab
make_taufigure()

# ╔═╡ 2f2b8ea0-02cf-4aeb-a132-8c5853d30910
with_theme(bw_theme) do
	lstyles = [(:dot, n) for n in range(2, 10, length = 10)]
	make_taufigure(linestyles = lstyles)
end

# ╔═╡ 5e36f9c7-d48e-4a3d-b45e-1e52063a6c81
"""
    m1qgrid(C, n)

Alias for `m1tauvals(C, n)`, providing a grid of `q` values.

**Arguments**
- `C`: A parameter used in the `tau_anal` function.
- `n`: An integer, passed to `m1tauvals`.

**Returns**
- An array of `q` values, equivalent to `tau` values.
"""
m1qgrid(C, n) = m1tauvals(C, n)

# ╔═╡ 6172f6b3-fe74-4538-9fcf-e3561ee52404
"""
    m1qvals(C, n)

Computes `q` values by applying `Q_anal` to each element in `ts`.

**Arguments**
- `C`: A parameter used in the `Q_anal` function.
- `n`: An integer, though not used in this function.

**Returns**
- An array of `q` values computed by `Q_anal`.
"""
m1qvals(C, n) = Q_anal.(m2ts(C, n), C)

# ╔═╡ 8d1d75e3-f7c8-46d0-ab08-c74b75d1b9f7
make_qtaufig(; linestyles = (:dash, :solid)) = let
	fig = Figure()
	ax = Axis(fig[1, 1])
	
	# ax1 = Axis(fig[1, 1], yaxisposition = :right)
	# hidespines!(ax1)
	# hidexdecorations!(ax1)
	# ax1.ylabel = L"τ"

	ax.xlabel = L"\theta"
	ax.ylabel = L"q,~\tau"

	highestC = argmax(p->p.C, model_parameters).C

	for (series_ind, (; M, C)) in enumerate(model_parameters)
		@assert M in (1, 2)

		color = Cycled(series_ind)
		linewidth = M == 1 ? 2.0 : 1.0

		linestyle = if linestyles[1] isa Symbol
			linestyles[1]
		else
			linestyles[1][series_ind]
		end
		
		taugrid = if M == 1
			m1taugrid(C, plotting_ngrid_points) 
		elseif M == 2
			grd = m2taugrid(m2sols[C])
			smooth(grd)
		end

		tauvals = if M == 1
			m1tauvals(C, plotting_ngrid_points)
		elseif M == 2
			m2tauvals(m2sols[C])
		end

		lines!(ax, taugrid, tauvals, 
			label = L"\tau: M=%$M, C=%$C", linestyle=linestyle, color = color, linewidth = linewidth)

		linestyle = if highestC == C
			:solid
		elseif linestyles[2] isa Symbol
			linestyles[2]
		else
			linestyles[2][series_ind]
		end

		qgrid = if M == 1
			m1qgrid(C, plotting_ngrid_points) 
		elseif M == 2
			grd = m2qgrid(m2sols[C], plotting_ngrid_points)
			smooth(grd)
		end

		qvals = if M == 1
			m1qvals(C, plotting_ngrid_points)
		elseif M == 2
			m2qvals(m2sols[C], plotting_ngrid_points)
		end

		lines!(ax, qgrid, qvals, label = L"q: M=%$M, C=%$C", color = color, linewidth = linewidth, linestyle = linestyle)
		if C == highestC
			lines!([0; qgrid[begin]], [0; qvals[begin]], label = L"q: M=%$M, C=%$C", color = color, linewidth = linewidth, linestyle = :solid)
			lines!([qgrid[end]; 1], [qvals[end]; 1], label = L"q: M=%$M, C=%$C", color = color, linewidth = linewidth, linestyle = :solid)
		end
	end

	axislegend(ax, position = :lt)
	fig
end

# ╔═╡ a781ae41-27f2-4774-8307-03d1465701bd
make_qtaufig()

# ╔═╡ 80f71efd-1205-484a-99b0-201278a7b2fd
with_theme(bw_theme) do
	lstyles1 = [(:dot, n) for n in range(2, 10, length = 10)]
	lstyles2 = [(:dashdot, n) for n in range(4, 10, length = 10)]
	make_qtaufig(linestyles = (lstyles1, lstyles2))
end

# ╔═╡ c15b1d6d-7473-44b1-8829-d12ca843a11f
make_qfigure(;linestyles = :solid) = let
	fig = Figure()
	ax = Axis(fig[1, 1])

	ax.ylabel = L"q"
	ax.xlabel = L"\theta"

	highestC = argmax(p->p.C, model_parameters).C

	for (series_ind, (; M, C)) in enumerate(model_parameters)
		
		color = Cycled(series_ind)
		linewidth = M == 1 ? 2.0 : 1.0
		
		qgrid = if M == 1
			m1qgrid(C, plotting_ngrid_points) 
		elseif M == 2
			grd = m2qgrid(m2sols[C], plotting_ngrid_points)
			smooth(grd)
		end

		qvals = if M == 1
			m1qvals(C, plotting_ngrid_points)
		elseif M == 2
			m2qvals(m2sols[C], plotting_ngrid_points)
		end

		linestyle = if highestC == C
			:solid
		elseif linestyles isa Symbol
			linestyles
		else
			linestyles[series_ind]
		end

		lines!(ax, qgrid, qvals, label = L"M=%$M, C=%$C", linewidth=linewidth, color = color, linestyle = linestyle)
		if C == highestC
			lines!([0; qgrid[begin]], [0; qvals[begin]], label = L"M=%$M, C=%$C", color = color, linewidth = linewidth, linestyle = linestyle)
			lines!([qgrid[end]; 1], [qvals[end]; 1], label = L"M=%$M, C=%$C", color = color, linewidth = linewidth, linestyle = linestyle)
		end
	end

	lines!(ax, [1/2, 1.], [0, 1], color = :black, linestyle=:dash)

	axislegend(ax, position = :lt, merge=true)
	fig
end

# ╔═╡ 41d1f80f-5c69-48cc-8da9-a29385af12c6
make_qfigure()

# ╔═╡ 6e16ab1b-554c-48a7-8f42-bd34e9c913ec
with_theme(bw_theme) do
	lstyles = [(:dashdot, n) for n in range(4, 10, length = 10)]
	make_qfigure(linestyles = lstyles)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Accessors = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
SummationByPartsOperators = "9f78cca6-572e-554e-b819-917d2f1cf240"

[compat]
Accessors = "~0.1.42"
CairoMakie = "~0.13.4"
ColorSchemes = "~3.29.0"
DataFrames = "~1.7.0"
Interpolations = "~0.15.1"
Latexify = "~0.16.7"
NLsolve = "~4.5.1"
PlutoUI = "~0.7.23"
Roots = "~2.2.7"
SummationByPartsOperators = "~0.5.75"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "102e18cb6d8465deeb3983a44b2aea2188f20cb3"

[[deps.ADTypes]]
git-tree-sha1 = "e2478490447631aedba0823d4d7a80b2cc8cdb32"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.14.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgCheck]]
git-tree-sha1 = "680b3b8759bd4c54052ada14e52355ab69e07876"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AutoHashEquals]]
git-tree-sha1 = "4ec6b48702dacc5994a835c1189831755e4e76ef"
uuid = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
version = "2.2.0"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "c1c90ea6bba91f769a8fc3ccda802e96620eb24c"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.13.4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2ac646d71d0d24b44f3f8c84da8c9f4d70fb67df"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.4+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "e771a63cc8b539eca78c85b0cabd9233d6c8f06f"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "5620ff4ee0084a6ab7097a27ba0c19290200b037"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.4"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "aa87a743e3778d35a950b76fbd2ae64f810a2bb3"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.6.52"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "6d8b535fd38293bc54b88455465a1386f8ac1c3c"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.119"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "e7b7e6f178525d17c720ab9c081e4ef04429f860"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.4"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.Extents]]
git-tree-sha1 = "063512a13dbe9c40d999c439268539aa552d1ae6"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "7de7c78d681078f027389e067864a8d53bd7c3c9"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "fd923962364b645f3719855c88f7074413a6ad92"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.0.2"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "f089ab1f834470c525562030c8cfde4025d5e915"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.27.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "d52e255138ac21be31fa633200b65e4e71d26802"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.6"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "8e233d5167e63d708d41f87597433f59a0f213fe"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.4"

[[deps.GeoInterface]]
deps = ["DataAPI", "Extents", "GeoFormatTypes"]
git-tree-sha1 = "294e99f19869d0b0cb71aef92f19d03649d028d5"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.4.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "65e3f5c519c3ec6a4c59f4c3ba21b6ff3add95b0"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.7"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "dc6bed05c15523624909b3953686c5f5ffa10adc"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8e070b599339d622e9a081d17230d74a5c473293"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.17"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InlineStrings]]
git-tree-sha1 = "6a9fde685a7ac1eb3495f8e812c5a7c3711c2d5e"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.3"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "LinearAlgebra", "MacroTools", "OpenBLASConsistentFPCSR_jll", "RoundingEmulator"]
git-tree-sha1 = "2c337f943879911c74bb62c927b65b9546552316"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.29"
weakdeps = ["DiffRules", "ForwardDiff", "IntervalSets", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.Jieko]]
deps = ["ExproniconLite"]
git-tree-sha1 = "2f05ed29618da60c06a87e9c033982d4f71d0b6c"
uuid = "ae98c720-c025-4a4a-838c-29b094483192"
version = "0.2.1"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd10d2cc78d34c0e2a3a36420ab607b611debfbb"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.7"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "36c4b9df1d1bac2fadb77b27959512ba6c541d91"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "8084c25a250e00ae427a379a5b607e7aed96a2dd"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.171"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "0318d174aa9ec593ddf6dc340b434657a8f1e068"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.22.4"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "903ef1d9d326ebc4a9e6cf24f22194d8da022b50"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.9.2"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "f5a6805fb46c0285991009b526ec6fae43c6dec2"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.Moshi]]
deps = ["ExproniconLite", "Jieko"]
git-tree-sha1 = "453de0fc2be3d11b9b93ca4d0fddd91196dcf1ed"
uuid = "2e0e35c7-a2e4-4343-998d-7ef72827ed2d"
version = "0.3.5"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "b14c7be6046e7d48e9063a0053f95ee0fc954176"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.9.1"

[[deps.NLsolve]]
deps = ["Distances", "LineSearches", "LinearAlgebra", "NLSolversBase", "Printf", "Reexport"]
git-tree-sha1 = "019f12e9a1a7880459d0173c182e6a99365d7ac1"
uuid = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
version = "4.5.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "567515ca155d0020a45b05175449b499c63e7015"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.29+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a9697f1d06cc3eb3fb3ad49cc67f2cfabaac31ea"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.16+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "0e1340b5d98971513bddaa6bbed470670cebbbfe"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.34"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "cf181f0b1e6a18dfeb0ee8acc4a9d1672499626c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.4"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "44f6c1f38f77cafef9450ff93946c53bd9ca16ff"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "5152abbdab6488d5eec6a01029ca6697dff4ec8f"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.23"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PolynomialBases]]
deps = ["ArgCheck", "AutoHashEquals", "FFTW", "FastGaussQuadrature", "LinearAlgebra", "Requires", "SimpleUnPack", "SpecialFunctions"]
git-tree-sha1 = "b62fd0464edfffce54393cd617135af30fa47006"
uuid = "c74db56a-226d-5e98-8bb0-a6049094aeea"
version = "0.4.22"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "13c5103482a8ed1536a54c08d0e742ae3dca2d42"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.4"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "112c876cee36a5784df19098b55db2b238afc36a"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.31.2"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "3ac13765751ffc81e3531223782d9512f6023f71"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.7"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "04c968137612c4a5629fa531334bb81ad5680f00"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.13"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "fea870727142270bdf7624ad675901a1ee3b4c87"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.1"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Moshi", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "f2db9ab9d33130df3be35be9438da51a3416d528"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.84.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMLStyleExt = "MLStyle"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "1c4b7f6c3e14e6de0af66e66b86d525cae10ecb4"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.13"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "566c4ed301ccb2a44cbd5a27da5f885e0ed1d5df"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleUnPack]]
git-tree-sha1 = "58e6353e72cde29b90a69527e56df1b5c3d8c437"
uuid = "ce78b400-467f-4804-87d8-8f486da07d0a"
version = "1.1.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "f737d444cb0ad07e61b3c1bef8eb91203c321eff"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.2.0"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "35b09e80be285516e52c9054792c884b9216ae3c"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.4.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "8ad2e38cbb812e29348719cc63580ec1dfeb9de4"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.1"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.SummationByPartsOperators]]
deps = ["ArgCheck", "AutoHashEquals", "FFTW", "InteractiveUtils", "LinearAlgebra", "LoopVectorization", "MuladdMacro", "PolynomialBases", "PrecompileTools", "RecursiveArrayTools", "Reexport", "Requires", "SciMLBase", "SimpleUnPack", "SparseArrays", "StaticArrayInterface", "StaticArrays", "Unrolled"]
git-tree-sha1 = "e9c8882c820eab4fdc50bcbbe79cb23b96692ca3"
uuid = "9f78cca6-572e-554e-b819-917d2f1cf240"
version = "0.5.75"

    [deps.SummationByPartsOperators.extensions]
    SummationByPartsOperatorsBandedMatricesExt = "BandedMatrices"
    SummationByPartsOperatorsDiffEqCallbacksExt = "DiffEqCallbacks"
    SummationByPartsOperatorsForwardDiffExt = "ForwardDiff"
    SummationByPartsOperatorsOptimForwardDiffExt = ["Optim", "ForwardDiff"]
    SummationByPartsOperatorsStructArraysExt = "StructArrays"

    [deps.SummationByPartsOperators.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    DiffEqCallbacks = "459566f4-90b8-5000-8ac3-15dfb0a30def"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "PrettyTables", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "7530e17b6ac652b009966f8ad53371a4ffd273f2"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.39"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "18ad3613e129312fe67789a71720c3747e598a61"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.3"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "f21231b166166bebc73b99cea236071eb047525b"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.3"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.Unrolled]]
deps = ["MacroTools"]
git-tree-sha1 = "6cc9d682755680e0f0be87c56392b7651efc2c7b"
uuid = "9602ed7d-8fef-5bc8-8597-8f21381861e8"
version = "0.1.5"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "4ab62a49f1d8d9548a1c8d1a75e5f55cf196f64e"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.71"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "068dfe202b0a05b8332f1e8e6b4080684b9c7700"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.47+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "d2408cac540942921e7bd77272c32e58c33d8a77"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.5.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╠═28b75488-9997-42e0-b2d5-e69a75960fce
# ╠═4671adce-b2f3-4834-b1d2-d9769fdea976
# ╠═5f09efcb-a077-436e-8e93-b78d0b47581d
# ╠═4390fb37-e5de-47d4-a61a-3ee01e833a9c
# ╟─6b1c78f2-a53d-40b7-b4db-57d4c4798e11
# ╟─31fbf75d-44cd-4ce2-a38d-5ccc8b6b3973
# ╟─06602b0a-34ff-499a-8c36-7d575b49fb3c
# ╟─722ec8c2-7a95-4616-8385-3e2c7f29c47a
# ╠═8027abff-2405-419b-a3c8-77f7b8a63c1a
# ╠═bb6a3d85-137c-43c9-b1df-bc777274dcda
# ╠═70ac10e0-7272-4f27-87fc-035b387e51bb
# ╠═f7b6b26c-ba00-4cd7-998c-c3f802d3a9aa
# ╟─30b77702-548c-4ec7-9b8c-5d3f8a979f5f
# ╠═40779d47-cddc-49a6-9a11-46f494853072
# ╟─cb1de805-72d8-4a44-9027-2f15e5115966
# ╟─9dfeb033-96e9-46e9-99a5-26cfa556242e
# ╠═3d4f6b18-a282-4462-a715-63bc933dce15
# ╠═10732e49-b440-442f-9bf0-b588704d59c9
# ╠═c88775ab-2c55-427c-b8b8-8c0f0bd5cc78
# ╠═fa64da8f-f3f6-4122-a5dc-044f4057f0d1
# ╠═975071ab-719f-4bd9-a57a-14a6c4d3d648
# ╠═758b14d5-71b0-4bf0-acf0-5dc525346488
# ╟─a2952c8c-425b-4e05-a379-8dbfb20b7708
# ╟─63df4212-f263-4fcb-9438-aa5a1f2d4afd
# ╟─ec560f14-a8d3-4400-ae39-eae91cb79401
# ╠═8d1d75e3-f7c8-46d0-ab08-c74b75d1b9f7
# ╠═a781ae41-27f2-4774-8307-03d1465701bd
# ╠═80f71efd-1205-484a-99b0-201278a7b2fd
# ╟─e24dd6d9-55ca-47a6-a8e4-746d54a794d2
# ╠═29b430c7-e482-4718-93fc-2e72e03f036d
# ╠═93141f29-0b47-4353-8241-b9ca791448ab
# ╠═2f2b8ea0-02cf-4aeb-a132-8c5853d30910
# ╟─19e4055f-b0ce-4fc7-a87d-9ec635af5d18
# ╠═c15b1d6d-7473-44b1-8829-d12ca843a11f
# ╠═41d1f80f-5c69-48cc-8da9-a29385af12c6
# ╠═6e16ab1b-554c-48a7-8f42-bd34e9c913ec
# ╟─758e9da4-4c63-460f-83ea-75b19cc9bc4e
# ╠═9174c5e2-3e33-4773-b791-35da82177784
# ╟─07785a81-04e2-4467-a94a-e72d7ba4ed90
# ╠═ffa31d71-600c-4a9c-9bb8-812f7e16e7e7
# ╟─543f96e6-702a-471f-bdaf-ac7fa1e78cac
# ╠═e896c5c6-876e-4ae5-9bf2-0f264baa5361
# ╟─1293036b-438e-44a7-bf98-5a801c2e7980
# ╠═d53543cd-069a-4295-bd64-74cb09e7dd46
# ╟─b186742f-27dc-44fb-845d-a8c0e88f1c31
# ╠═d9e9a00c-d7b5-4067-ab6e-14a43360e9b0
# ╠═0bcb2986-918c-4122-aef7-d50af87369a4
# ╟─bb680f62-a993-454e-a9d4-afdce50ce401
# ╠═f7e54217-0bf9-4a2e-8485-46e7347fa05c
# ╟─cbfe6d7d-0448-4810-b823-e53cbbe31572
# ╠═8d352d4d-0e7f-4e67-aa54-4a425a7f95db
# ╟─11af02ed-d7ea-47df-875c-f7b3b0577847
# ╟─e825a420-fb13-40c8-8e67-5c41306529d0
# ╠═a8d7b84a-603b-4877-b3a9-c0d63a098e85
# ╠═92ebc267-984a-415d-b4ab-12fa187de59a
# ╠═a66b8e4a-794f-4978-b973-51eb59869d42
# ╠═76594335-40ce-4d19-95c0-1be236ca5aa7
# ╠═2c6db510-9afa-4081-b1b6-64bb07245460
# ╟─dd044433-8e5f-4ad2-a0e4-f82cc302c575
# ╟─69d474af-b030-444c-bff2-4841cd5f4ed1
# ╠═03372f88-1c7d-48f5-b73d-a754b4b5bb49
# ╠═2c7afc7e-d1a2-4fa4-ae0d-400fb1b529ed
# ╟─6129104e-d352-4b98-81fa-8351e32c0cdb
# ╠═0f39d1fc-3dff-4dfd-bb69-593e5d6e36c9
# ╟─f7262323-4401-4fc1-aeb8-ea4e70f86e98
# ╠═993ecf93-0d57-41f1-a0cb-a1b4577383f5
# ╠═82b88c4d-633a-4530-97b1-3103f2f59bc8
# ╠═d10f2a6b-2479-4f71-ad04-6ee787f99184
# ╠═0e93b2a9-6e81-497c-8248-98c096d3bbe5
# ╠═75646561-783c-4de2-b529-78c362628b19
# ╟─66cc35ca-b110-4cc9-807f-5d31e89004ef
# ╠═f9fdf572-fea0-4f8a-83dc-26b6794a95e4
# ╠═303ab0c2-e951-4901-a83e-da81844d3f25
# ╠═a6c23fbf-abea-4144-9ef4-4d26584d0404
# ╠═1ac0fee2-585f-45db-bce9-f615756f03ab
# ╠═716e84f8-289e-46a3-bb1e-48e3fe4ca5a0
# ╠═836a2c06-8cab-4f60-8895-36fc2207bfa2
# ╠═602329dd-a9e5-4106-a36e-edbf26316a9b
# ╠═c9c315bd-63fa-4fac-99a2-42f5a5301709
# ╠═3f986b17-fa96-4e2b-a61d-378535ace104
# ╠═50dddf04-dc51-41d3-97b4-35034990fbb2
# ╠═5e36f9c7-d48e-4a3d-b45e-1e52063a6c81
# ╠═6172f6b3-fe74-4538-9fcf-e3561ee52404
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
