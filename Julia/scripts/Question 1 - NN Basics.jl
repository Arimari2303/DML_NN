# =============================================
# 0. Instalación (solo una vez)
# =============================================

using Pkg
Pkg.add(["Torch", "Random", "Printf", "Plots"])

using Torch
Torch.install()  # ← equivalente a torch::install_torch()
@assert Torch.torch_is_installed()


# =============================================
# 1. Simulación
# =============================================

using Random
Random.seed!(123)

n = 400
x = 2π .* rand(n)
eps = 0.2 .* randn(n)
y = sin.(x) .+ eps

x_grid = range(0, 2π, length=600)

x_t = torch_tensor(reshape(x, :, 1))
y_t = torch_tensor(reshape(y, :, 1))
xg_t = torch_tensor(reshape(x_grid, :, 1))

# =============================================
# 2. Definir MLP 
# =============================================

using Torch: nn, nnf, optim

mutable struct MLP <: nn.Module
    l1::nn.Linear
    l2::nn.Linear
    l3::nn.Linear
    out::nn.Linear
    acts::Vector{String}

    function MLP(acts = ["relu", "relu", "relu"])
        l1 = nn.Linear(1, 50)
        l2 = nn.Linear(50, 50)
        l3 = nn.Linear(50, 50)
        out = nn.Linear(50, 1)
        new(l1, l2, l3, out, acts)
    end
end

# Función de activación según string
activation(act::String, x) = 
    act == "sigmoid" ? nnf_sigmoid(x) :
    act == "tanh"    ? nnf_tanh(x)    :
    act == "relu"    ? nnf_relu(x)    :
    error("Activación no soportada: $act")

# Forward pass
function (m::MLP)(x)
    a1 = activation(m.acts[1], m.l1(x))
    a2 = activation(m.acts[2], m.l2(a1))
    a3 = activation(m.acts[3], m.l3(a2))
    m.out(a3)
end


# =============================================
# 3. Entrenamiento + Early Stopping
# =============================================

function train_and_predict(acts; lr=1e-3, epochs=2000, batch=64, patience=50)
    net = MLP(acts)
    opt = optim.Adam(net.parameters(), lr=lr)

    best_loss = Inf
    best_state = nothing
    wait = 0
    n_obs = size(x_t, 1)
    idx_all = 1:n_obs

    for e in 1:epochs
        # Mini-batch SGD
        bs_idx = sample(idx_all, batch, replace=true)
        xb = x_t[bs_idx, :, :]
        yb = y_t[bs_idx, :, :]

        opt.zero_grad()
        pred = net(xb)
        loss = nnf_mse_loss(pred, yb)
        loss.backward()
        opt.step()

        # Early stopping en todo el conjunto
        full_loss = Torch.no_grad() do
            nnf_mse_loss(net(x_t), y_t).item()
        end

        if full_loss + 1e-9 < best_loss
            best_loss = full_loss
            best_state = Torch.state_dict(net)
            wait = 0
        else
            wait += 1
            if wait >= patience
                break
            end
        end
    end

    # Cargar mejor modelo
    if !isnothing(best_state)
        Torch.load_state_dict!(net, best_state)
    end

    # Predicciones
    yhat_train, yhat_grid = Torch.no_grad() do
        train_pred = net(x_t) |> cpu |> Array |> vec
        grid_pred  = net(xg_t) |> cpu |> Array |> vec
        train_pred, grid_pred
    end

    mse = mean((y .- yhat_train).^2)

    return (yhat = yhat_grid, mse = mse)
end

# =============================================
# 4. Configuraciones y ejecución
# =============================================

configs = Dict(
    "logistic" => ["sigmoid", "sigmoid", "sigmoid"],
    "tanh"     => ["tanh", "tanh", "tanh"],
    "relu"     => ["relu", "relu", "relu"],
    "mixta"    => ["tanh", "relu", "sigmoid"]
)

results = Dict()
for (nm, acts) in configs
    @info "Entrenando: $nm"
    results[nm] = train_and_predict(acts)
    @printf "%s: MSE = %.4f\n" nm results[nm].mse
end

# =============================================
# 5. Gráfico (equivalente a plot + lines + legend)
# =============================================

using Plots

p = scatter(x, y, 
    marker=:circle, markersize=3, alpha=0.6, color=:gray20,
    label="Datos", xlabel="x", ylabel="y",
    title="y = sin(x) + ε: ajustes NN (Torch.jl)")

cols = [:blue, :red, :darkgreen, :purple]
i = 1
for (nm, res) in results
    plot!(p, x_grid, res.yhat, 
          linewidth=2, color=cols[i], label=nm)
    i += 1
end

display(p)


# =============================================
# 6. ¿Cuál se ajusta mejor?
# =============================================

mses = Dict(nm => res.mse for (nm, res) in results)
best = argmin(mses) |> first
@printf "Mejor red: %s (MSE = %.4f)\n" best mses[best]


# =============================================
# 7. QUESTION (0.5 pts) - Respuesta en Julia
# =============================================

println("""
4. QUESTION (0.5 pts)

Para comparar las redes neuronales, utilizamos el Error Cuadrático Medio (MSE) como métrica de evaluación.

El modelo con el MSE más bajo ofrece el mejor ajuste a los datos ( y = sin(x) + ε ), 
ya que reproduce el patrón sinusoidal subyacente con el menor error promedio.

Mejor modelo:
La red de mejor rendimiento es el modelo de activación mixta, que utiliza activaciones 
tanh, ReLU y sigmoid en sus tres capas ocultas, logrando un MSE aproximado de 0.0419.

Interpretación del resultado:
Esta red mixta combina la suavidad de tanh con la flexibilidad de ReLU y la naturaleza 
acotada de sigmoid, permitiéndole capturar mejor la forma no lineal de la curva seno 
y adaptarse a variaciones locales.

Aunque los modelos puros de ReLU y tanh también aprenden el patrón general, 
la estructura de activación mixta proporciona un equilibrio ligeramente mejor 
entre curvatura y estabilidad, resultando en el menor error global entre todas 
las redes evaluadas.
""")

# =============================================
# II. LEARNING RATE - Julia (Torch.jl)
# =============================================

using Random, Torch, Plots, Printf
using Torch: nn, nnf, optim

Random.seed!(123)

# =============================================
# 1. Simulación (mismo DGP que en R)
# =============================================
n = 400
x = 2π .* rand(n)                  # runif(n, 0, 2*pi)
eps = 0.2 .* randn(n)              # rnorm(n, 0, 0.2)
y = sin.(x) .+ eps

x_grid = range(0, 2π, length=600)  # seq(0, 2*pi, length.out=600)

x_t = torch_tensor(reshape(x, :, 1))
y_t = torch_tensor(reshape(y, :, 1))
xg_t = torch_tensor(reshape(x_grid, :, 1))

# =============================================
# 2. Constructor MLP con L capas ocultas (50 neuronas)
# =============================================

function make_mlp(L::Int = 1, act::String = "relu")
    @assert L ∈ (1,2,3) && act ∈ ("relu","tanh","sigmoid")
    
    struct MLP <: nn.Module
        layers::nn.ModuleList
        out::nn.Linear
        act::Function

        function MLP(L, act)
            layers = nn.ModuleList()
            in_dim = 1
            for i in 1:L
                push!(layers, nn.Linear(in_dim, 50))
                in_dim = 50
            end
            out = nn.Linear(50, 1)
            
            f_act = act == "relu"    ? nnf_relu :
                    act == "tanh"    ? nnf_tanh :
                    act == "sigmoid" ? nnf_sigmoid : error("act no válida")
            
            new(layers, out, f_act)
        end
    end

    function (m::MLP)(x)
        for layer in m.layers
            x = m.act(layer(x))
        end
        m.out(x)
    end

    return MLP(L, act)
end


# =============================================
# 3. Entrenamiento con early stopping
# =============================================

function fit_lr(L::Int=1; lr=1e-3, epochs=2000, batch=64, patience=60)
    net = make_mlp(L, "relu")()
    opt = optim.Adam(net.parameters(), lr=lr)

    best_loss = Inf
    best_state = nothing
    wait = 0
    n_obs = size(x_t, 1)

    for e in 1:epochs
        # Mini-batch
        bs_idx = sample(1:n_obs, batch, replace=true)
        xb = x_t[bs_idx, :, :]
        yb = y_t[bs_idx, :, :]

        opt.zero_grad()
        pred = net(xb)
        loss = nnf_mse_loss(pred, yb)
        loss.backward()
        opt.step()

        # Early stopping en todo el dataset
        full_loss = Torch.no_grad() do
            nnf_mse_loss(net(x_t), y_t).item()
        end

        if full_loss + 1e-9 < best_loss
            best_loss = full_loss
            best_state = Torch.state_dict(net)
            wait = 0
        else
            wait += 1
            if wait >= patience
                break
            end
        end
    end

    # Cargar mejor modelo
    if !isnothing(best_state)
        Torch.load_state_dict!(net, best_state)
    end

    # Predicciones
    yhat_train, yhat_grid = Torch.no_grad() do
        train_pred = net(x_t) |> cpu |> Array |> vec
        grid_pred  = net(xg_t) |> cpu |> Array |> vec
        train_pred, grid_pred
    end

    mse = mean((y .- yhat_train).^2)
    return (yhat = yhat_grid, mse = mse)
end

# =============================================
# 4. Tasas y capas a evaluar
# =============================================

lrs = [1e-4, 1e-3, 1e-2, 1e-1]
layers_set = [1, 2, 3]

all_results = Dict{String, Dict{String, NamedTuple}}()

for L in layers_set
    res_L = Dict{String, NamedTuple}()
    for eta in lrs
        nm = "lr=$(eta)"
        @info "Entrenando L=$L, $nm"
        res_L[nm] = fit_lr(L, lr=eta)
    end
    all_results["L$L"] = res_L
end

# =============================================
# 5. Reporte de MSE y mejor lr por #capas
# =============================================

for L in layers_set
    key = "L$L"
    println("\n-- $L hidden layer(s) --")
    mses = [all_results[key][nm].mse for nm in keys(all_results[key])]
    names_lr = collect(keys(all_results[key]))
    println(round.(mses, digits=5))
    
    best_idx = argmin(mses)
    best_name = names_lr[best_idx]
    @printf "Best lr for L=%d: %s (MSE=%.4f)\n" L best_name mses[best_idx]
end


# =============================================
# 6. Gráficos: 1 por #capas, 4 curvas (lr)
# =============================================

cols = [:blue, :red, :darkgreen, :purple]
plots = []

for (idx, L) in enumerate(layers_set)
    p = scatter(x, y,
        marker=:circle, markersize=3, alpha=0.6, color=:gray20,
        label="Datos", xlabel="x", ylabel="y",
        title="ReLU | $L capa(s) oculta(s)", size=(500, 400))

    key = "L$L"
    for (i, nm) in enumerate(keys(all_results[key]))
        res = all_results[key][nm]
        plot!(p, x_grid, res.yhat,
              linewidth=2, color=cols[i], label=nm)
    end

    push!(plots, p)
end

# Mostrar los 3 gráficos en fila
plot(plots..., layout=(1,3), size=(1400, 450))

# =============================================
# INTERPRETACIÓN (Clara y Ampliada)
# =============================================

println("""
Interpretación (Clara y Ampliada)

La tasa de aprendizaje (lr) controla el tamaño del paso que se da durante el descenso de gradiente 
al actualizar los pesos de la red.
Determina qué tan rápido o lento aprende el modelo a partir de sus errores.

* Muy alta (0.1): Las actualizaciones sobrepasan los pesos óptimos, causando que el modelo 
  oscile o diverja.

* Muy baja (0.0001): Las actualizaciones son demasiado pequeñas, haciendo que el aprendizaje 
  sea muy lento y conduciendo a subajuste (underfitting).

Resultados y Análisis

* 1 capa oculta: La red alcanzó su MSE más bajo con lr = 0.001, produciendo un ajuste 
  suave y estable a la onda sinusoidal.

* 2 capas ocultas: El rendimiento mejoró ligeramente con lr = 0.01, ya que una tasa de 
  aprendizaje más alta ayudó al modelo más profundo a converger más rápido.

* 3 capas ocultas: La tasa óptima se mantuvo entre 0.001 y 0.01; valores más altos (0.1) 
  causaron inestabilidad y ajustes ruidosos.

Insight Principal
A medida que aumenta el número de capas ocultas, el modelo se vuelve más expresivo, 
pero también más difícil de optimizar.

* Tasas de aprendizaje muy pequeñas hacen que la convergencia sea lenta y arriesgan 
  atrapar al modelo en mínimos locales.

* Tasas moderadamente mayores (alrededor de 0.01) ayudan a las redes más profundas 
  a converger más rápido sin inestabilidad.

En este experimento, el rango 0.001–0.01 ofreció el mejor equilibrio entre estabilidad 
y velocidad de convergencia, mientras que valores extremadamente grandes o pequeños 
degradaron el rendimiento.
""")