import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import gurobipy as gp
from gurobipy import GRB

pd.set_option('display.max_columns', None)

customers_df = pd.read_excel("/Users/serhandulger/Desktop/MODELING_AND_OPTIMIZATION/DRAFT_V1/ev_customers_df_cleaned_proportional.xlsx")

distance_matrix_df = pd.read_csv("/Users/serhandulger/Desktop/MODELING_AND_OPTIMIZATION/DRAFT_V1/ev_distance_matrix_df.csv")

def checking_df(df):
    print("### FIRST FIVE ROWS ###")
    print(df.head())
    print("### DATAFRAME ROWS/ FEATURES COUNT ###")
    print(df.shape)
    print("### DATAFRAME FEATURES DATA TYPES ###")
    print(df.dtypes)
    print("### NULL VALUES COUNT ###")
    print(df.isnull().sum())

checking_df(customers_df)

df = customers_df.copy()

## Model preprocessing: standardizing missing values to ensure stable value score calculations
df["repurchase_prob"] = df["repurchase_prob"].fillna(0)
df["ltv"] = df["ltv"].fillna(0)
df["acquisition_prob"] = df["acquisition_prob"].fillna(0)

########################################################################################################################

## Segment-based base pricing (fixed to isolate discount and premium effects for now - it may change depends on the business rules)
df["base_price"] = np.where(df["segment"].eq("own"), 0.50, 0.50)

########################################################################################################################

## LTV-based discount for own customers (higher LTV implies higher discount)
disc_max = 0.25  # maximum discount rate for own customers
df["own_discount"] = np.where(df["segment"].eq("own"),disc_max * (df["ltv"] / 100.0),0.0)

## Brand-premium-based surcharge for competitor customers
prem_max = 0.30  # premium intensity parameter
df["comp_premium"] = np.where(df["segment"].eq("competitor"),prem_max * (df["brand_premium"] - 1.0),0.0)

########################################################################################################################

## Adjusted price per kWh after applying discounts and premiums
df["price_adj"] = df["base_price"] * (1 - df["own_discount"]) * (1 + df["comp_premium"])

## Adjusted revenue computation: kWh demand multiplied by adjusted price
df["revenue_adj"] = (df["kwh_needed"] * df["price_adj"]).round(2)

df[["request_id", "segment", "ltv", "brand_premium","base_price", "own_discount", "comp_premium","price_adj", "kwh_needed", "revenue_adj"]].head(5)

## Short-term net contribution: adjusted revenue minus travel cost
cost_per_km = 0.35  # operational cost per kilometer (labor, energy, logistics)
df["travel_cost"] = df["distance_from_depot_km"] * cost_per_km

df["short_term_net"] = df["revenue_adj"] - df["travel_cost"]
## Negative values indicate requests that are unprofitable in the short run and require further inspection

########################################################################################################################

## Long-term component for own customers: loyalty and repurchase effect
## Only applicable to own customers
df["long_term_own"] = np.where(df["segment"] == "own",df["ltv"] * df["repurchase_prob"],0.0)

########################################################################################################################

## Long-term component for competitor customers: acquisition potential
expected_ltv_if_acquired = 60.0  # assumed average LTV after successful acquisition

df["long_term_comp"] = np.where(df["segment"] == "competitor",df["acquisition_prob"] * df["brand_premium"] * expected_ltv_if_acquired,0.0)

df[["request_id", "ltv", "segment", "service_time_min","price_adj", "revenue_adj", "kwh_needed","travel_cost", "short_term_net","long_term_own", "long_term_comp", "service_time_min"]].head(5)

########################################################################################################################

### ADDITIONAL NOTES ###

## Externally assumed parameters used in this step:
## - cost_per_km
## - repurchase_prob (customer-specific)
## - acquisition_prob
## - expected_ltv_if_acquired
## - traffic multiplier (introduced in the next block)

########################################################################################################################

# -----------------------------
# VALUE SCORE (input to the knapsack objective)
# -----------------------------

## Weight of short-term net contribution (dominant component)
w_short = 1.0

## Weight of long-term loyalty value for own customers
## Scaled down to reflect future (non-immediate) monetary impact
w_own = 0.20

## Weight of long-term acquisition value for competitor customers
## Can be set higher or lower depending on strategic priorities
w_comp = 0.20

## Aggregated value score to be maximized by the knapsack model
df["value_score"] = (w_short * df["short_term_net"]+ w_own * df["long_term_own"]+ w_comp * df["long_term_comp"])

## Enforce non-negativity to avoid selecting low-value or loss-making requests
df["value_score"] = df["value_score"].clip(lower=0)

df[["request_id", "ltv", "segment","short_term_net", "repurchase_prob","long_term_own", "long_term_comp","value_score"]].head(5)

# DEPOT → CUSTOMER TRAVEL TIME (minutes)

## Baseline assumption: average urban speed ≈ 40 km/h
## 60 minutes / 40 km ≈ 1.5 minutes per km (traffic-free reference)
MIN_PER_KM = 1.5  # minutes required to travel 1 km under normal conditions

## One-way travel time from depot to customer (minutes)
df["travel_time_min"] = (df["distance_from_depot_km"] * MIN_PER_KM * df["traffic_multiplier"])

# TOTAL CUSTOMER TIME (SERVICE + TRAVEL)

## service_time_min: on-site service duration
## travel_time_min : travel time from depot
## Their sum represents the true operational time cost per request
df["total_time_min"] = (df["service_time_min"] + df["travel_time_min"])

####################
# MODELLING
####################

ids = df["request_id"].tolist()

value   = dict(zip(df["request_id"], df["value_score"]))
service = dict(zip(df["request_id"], df["total_time_min"]))
energy  = dict(zip(df["request_id"], df["kwh_needed"]))

# Single-vehicle daily capacities
S_max = 460   # minutes
E_max = 225   # kWh

# -----------------------------
# Scenario settings
# -----------------------------
K_list = list(range(5, 31))
exact = True  # True: ==K, False: <=K

results = []

for K in K_list:
    m = gp.Model(f"knapsack_K{K}")
    m.setParam("OutputFlag", 0)  # keep silent

    x = m.addVars(ids, vtype=GRB.BINARY, name="x")

    # Objective: maximize total value_score
    m.setObjective(gp.quicksum(value[i] * x[i] for i in ids), GRB.MAXIMIZE)

    # Quota constraint: select exactly K (or at most K)
    if exact:
        m.addConstr(gp.quicksum(x[i] for i in ids) == K, name="quota_exact")
    else:
        m.addConstr(gp.quicksum(x[i] for i in ids) <= K, name="quota_max")

    # Capacity constraints
    m.addConstr(gp.quicksum(service[i] * x[i] for i in ids) <= S_max, name="time_budget")
    m.addConstr(gp.quicksum(energy[i]  * x[i] for i in ids) <= E_max, name="energy_budget")

    m.optimize()

    feasible = (m.SolCount > 0)

    if feasible:
        selected_ids = [i for i in ids if x[i].X > 0.5]
        sel_n = len(selected_ids)

        total_time   = sum(service[i] for i in selected_ids)
        total_energy = sum(energy[i]  for i in selected_ids)
        total_value  = sum(value[i]   for i in selected_ids)

        avg_time   = total_time / sel_n if sel_n > 0 else 0
        avg_energy = total_energy / sel_n if sel_n > 0 else 0

        results.append({
            "K_target": K,
            "feasible": True,
            "selected_n": sel_n,
            "total_value_score": total_value,
            "total_time_min": total_time,
            "time_util": total_time / S_max,
            "total_energy_kwh": total_energy,
            "energy_util": total_energy / E_max,
            "avg_time_min": avg_time,
            "avg_energy_kwh": avg_energy,
        })
    else:
        results.append({
            "K_target": K,
            "feasible": False,
            "selected_n": 0,
            "total_value_score": None,
            "total_time_min": None,
            "time_util": None,
            "total_energy_kwh": None,
            "energy_util": None,
            "avg_time_min": None,
            "avg_energy_kwh": None,
        })

res_df = pd.DataFrame(results)

print(res_df.to_string(index=False))

if exact:
    feasible_K = res_df.loc[res_df["feasible"], "K_target"]
    if len(feasible_K) > 0:
        print("\nMax feasible K (exact selection):", int(feasible_K.max()))
    else:
        print("\nNo feasible K found under exact selection.")
else:
    print("\nMax selected_n under <=K:", int(res_df["selected_n"].max()))


ids = df["request_id"].tolist()

value   = dict(zip(df["request_id"], df["value_score"]))
time_c  = dict(zip(df["request_id"], df["total_time_min"]))   # total operational time
energy  = dict(zip(df["request_id"], df["kwh_needed"]))

K = 13
S_max = 460   # minutes (daily time capacity)
E_max = 225   # kWh (daily energy capacity)

m = gp.Model("knapsack_selection_K13")
m.setParam("OutputFlag", 1)

x = m.addVars(ids, vtype=GRB.BINARY, name="x")

# Objective: maximize total value score
m.setObjective(gp.quicksum(value[i] * x[i] for i in ids), GRB.MAXIMIZE)

# Constraints
m.addConstr(gp.quicksum(x[i] for i in ids) == K, name="quota_exact_K")
m.addConstr(gp.quicksum(time_c[i] * x[i] for i in ids) <= S_max, name="time_budget")
m.addConstr(gp.quicksum(energy[i] * x[i] for i in ids) <= E_max, name="energy_budget")

m.optimize()

print("Status:", m.Status, "SolCount:", m.SolCount)

if m.SolCount > 0:
    selected_ids = [i for i in ids if x[i].X > 0.5]

    total_time = sum(time_c[i] for i in selected_ids)
    total_energy = sum(energy[i] for i in selected_ids)
    total_value = sum(value[i] for i in selected_ids)

    print("Selected:", len(selected_ids))
    print("Total time (min):", total_time, "/", S_max)
    print("Total energy (kWh):", total_energy, "/", E_max)
    print("Total value_score:", total_value)

    # Mark selection and create routing input
    df["selected"] = df["request_id"].isin(selected_ids).astype(int)
    selected_df = df[df["selected"] == 1].copy()   # <-- THIS is what you pass to TSP/VRP
else:
    print("No solution found (constraints may be too tight for this K).")
    selected_df = df.iloc[0:0].copy()

df["selected"] = df["request_id"].isin(selected_ids)

selected_df     = df[df["selected"]].copy()
not_selected_df = df[~df["selected"]].copy()

####################
# DISTANCE MATRIX
###################

distance_matrix_df.head()

# =========================
# 0) INPUTS (already defined in your notebook)
# =========================
edges = distance_matrix_df  # <-- your existing edge list dataframe
selected_ids = selected_df["request_id"].tolist()  # knapsack result
DEPOT = 0  # change if your depot id is different

# Quick column check
needed_cols = {"from_node_id", "to_node_id", "distance_km", "travel_time_min"}
missing = needed_cols - set(edges.columns)
if missing:
    raise ValueError(f"distance_matrix_df is missing columns: {missing}")

# =========================
# 1) BUILD NODE SET (depot + selected customers)
# =========================
nodes = [DEPOT] + selected_ids
nodes = list(dict.fromkeys(nodes))  # unique
n = len(nodes)

node_to_idx = {node: idx for idx, node in enumerate(nodes)}
idx_to_node = {idx: node for node, idx in node_to_idx.items()}

# =========================
# 2) FILTER EDGE LIST TO SUBGRAPH
# =========================
sub = edges[edges["from_node_id"].isin(nodes) & edges["to_node_id"].isin(nodes)].copy()

expected = n * (n - 1)  # all directed pairs without diagonal
if len(sub) != expected:
    raise ValueError(
        f"Subgraph is incomplete: expected {expected} directed edges, found {len(sub)}.\n"
        f"Check DEPOT id, selected_ids mapping, and whether the edge list contains all pairs."
    )

# =========================
# 3) BUILD TIME & DIST MATRICES (n x n)
# =========================
time_mat = np.zeros((n, n), dtype=float)
dist_mat = np.zeros((n, n), dtype=float)

for r in sub.itertuples(index=False):
    i = node_to_idx[getattr(r, "from_node_id")]
    j = node_to_idx[getattr(r, "to_node_id")]
    time_mat[i, j] = float(getattr(r, "travel_time_min"))
    dist_mat[i, j] = float(getattr(r, "distance_km"))

np.fill_diagonal(time_mat, 0.0)
np.fill_diagonal(dist_mat, 0.0)

# =========================
# 4) SOLVE TSP (Exact, Gurobi) - Minimize total travel time
# =========================
m = gp.Model("TSP_baseline_single_vehicle")
m.setParam("OutputFlag", 1)

x = m.addVars(n, n, vtype=GRB.BINARY, name="x")

# No self arcs
for i in range(n):
    m.addConstr(x[i, i] == 0, name=f"no_self_{i}")

# Degree constraints
for i in range(n):
    m.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1, name=f"out_{i}")
    m.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1, name=f"in_{i}")

# MTZ subtour elimination
u = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=n-1, name="u")
m.addConstr(u[0] == 0, name="u_depot")

for i in range(1, n):
    m.addConstr(u[i] >= 1, name=f"u_lb_{i}")
    m.addConstr(u[i] <= n-1, name=f"u_ub_{i}")

for i in range(1, n):
    for j in range(1, n):
        if i != j:
            m.addConstr(u[i] - u[j] + (n-1) * x[i, j] <= n-2, name=f"mtz_{i}_{j}")

m.setObjective(gp.quicksum(time_mat[i, j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
m.optimize()

if m.SolCount == 0:
    raise RuntimeError("TSP has no solution. (Unexpected if subgraph is complete.)")

# =========================
# 5) EXTRACT TOUR ORDER
# =========================
succ = {}
for i in range(n):
    for j in range(n):
        if x[i, j].X > 0.5:
            succ[i] = j

tour_idx = [0]
cur = 0
for _ in range(n - 1):
    cur = succ[cur]
    tour_idx.append(cur)
tour_idx.append(0)  # return depot

tour_nodes = [idx_to_node[i] for i in tour_idx]

total_time = sum(time_mat[a, b] for a, b in zip(tour_idx[:-1], tour_idx[1:]))
total_dist = sum(dist_mat[a, b] for a, b in zip(tour_idx[:-1], tour_idx[1:]))

print("\n=== TSP Baseline Results (Single Vehicle) ===")
print("Route (node ids):")
print(" -> ".join(map(str, tour_nodes)))
print(f"Total travel time (min): {total_time:.2f}")
print(f"Total distance (km):     {total_dist:.2f}")

# =========================
# 6) VISUALIZATION (2D embedding from distance matrix)
# =========================
def classical_mds(D, p=2):
    n0 = D.shape[0]
    D2 = D**2
    J = np.eye(n0) - np.ones((n0, n0)) / n0
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    L = np.diag(np.sqrt(np.maximum(eigvals[:p], 0)))
    V = eigvecs[:, :p]
    return V @ L

# Symmetrize for embedding
D_sym = 0.5 * (dist_mat + dist_mat.T)
coords = classical_mds(D_sym, p=2)

plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1])

for i in range(n):
    lbl = str(idx_to_node[i])
    if i == 0:
        plt.text(coords[i, 0], coords[i, 1], f"DEPOT({lbl})", fontweight="bold")
    else:
        plt.text(coords[i, 0], coords[i, 1], lbl)

for a, b in zip(tour_idx[:-1], tour_idx[1:]):
    x0, y0 = coords[a]
    x1, y1 = coords[b]
    plt.plot([x0, x1], [y0, y1])

plt.title("TSP Baseline Route (2D embedding from distance matrix)")
plt.xlabel("MDS dimension 1")
plt.ylabel("MDS dimension 2")
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# TSP REPORT (K=13)
# ------------------------------------------------------------
# Inputs:
# - selected_df: knapsack selected set
# - tsp_travel_time: TSP total travel time (min)

S_max = 460
tsp_travel_time = 135.85  # from TSP output

# --- Time components ---
service_time_sum = selected_df["service_time_min"].sum()
knapsack_time_sum = selected_df["total_time_min"].sum()          # selection-stage time proxy
route_total_time = service_time_sum + tsp_travel_time            # service + TSP travel (route-based)

time_diff = knapsack_time_sum - route_total_time                 # positive => knapsack proxy is more conservative
travel_share = tsp_travel_time / route_total_time

print("=== TIME SUMMARY (K=13) ===")
print(f"Daily time capacity (min):        {S_max:.0f}")
print(f"Service time total (min):         {service_time_sum:.2f}")
print(f"Route travel time total (min):    {tsp_travel_time:.2f}")
print(f"Route-based total time (min):     {route_total_time:.2f}  (util={route_total_time/S_max:.2%})")
print(f"Knapsack time proxy total (min):  {knapsack_time_sum:.2f}  (util={knapsack_time_sum/S_max:.2%})")
print(f"Proxy - route difference (min):   {time_diff:.2f}  (proxy is higher => more conservative)")
print(f"Travel share of route-based time: {travel_share:.2%}")

# Value / profit KPIs
total_value = selected_df["value_score"].sum()
value_per_min = total_value / route_total_time

print("\n=== VALUE SUMMARY (K=13) ===")
print(f"Total value_score:                {total_value:.2f}")
print(f"Value_score per minute:           {value_per_min:.4f}")

if "short_term_net" in selected_df.columns:
    total_short_net = selected_df["short_term_net"].sum()
    short_net_per_min = total_short_net / route_total_time
    print(f"Total short_term_net:             {total_short_net:.2f}")
    print(f"Short_term_net per minute:        {short_net_per_min:.4f}")


##################################################
# Clarke–Wright Savings VRP (depot-based, fixed V routes)
# Cost uses travel time (proposal: travel-time savings)
##################################################

S_max_per_vehicle = 460  # daily time capacity per vehicle

# --- service_time dict is required for route_total_time_min (travel + service) ---
service_time = {DEPOT: 0.0}
service_time.update(dict(zip(selected_df["request_id"], selected_df["service_time_min"])))

# selected customers only
customers = [c for c in selected_df["request_id"].tolist() if c != DEPOT]

DEPOT_NODE = DEPOT
depot_idx = node_to_idx[DEPOT_NODE]
cust_idx = [node_to_idx[c] for c in customers]  # customers are node-ids, convert to indices

def route_travel_time(route):
    return sum(time_mat[a, b] for a, b in zip(route[:-1], route[1:]))

def route_distance(route):
    return sum(dist_mat[a, b] for a, b in zip(route[:-1], route[1:]))

def route_service_time(route):
    cust_nodes = [idx_to_node[i] for i in route if idx_to_node[i] != DEPOT_NODE]
    return sum(service_time[n] for n in cust_nodes)

def compute_solution_metrics(routes_idx, S_max=460):
    route_metrics = []
    total_tt = 0.0
    total_km = 0.0

    for r in routes_idx:
        tt = route_travel_time(r)
        km = route_distance(r)
        st = route_service_time(r)
        total = tt + st
        cust_nodes = [idx_to_node[i] for i in r if idx_to_node[i] != DEPOT_NODE]

        route_metrics.append({
            "travel_time_min": tt,
            "distance_km": km,
            "service_time_min": st,
            "route_total_time_min": total,
            "customers": cust_nodes
        })

        total_tt += tt
        total_km += km

    max_route_total = max((rm["route_total_time_min"] for rm in route_metrics), default=0.0)
    feasible = (max_route_total <= S_max)

    return {
        "total_travel_time_min": total_tt,
        "total_distance_km": total_km,
        "max_route_total_time_min": max_route_total,
        "feasible_460min": feasible,
        "route_metrics": route_metrics
    }

def clarke_wright_vrp(V, cost_mat):
    """
    Clarke–Wright Savings heuristic:
    Start with routes [D,i,D], merge routes based on savings until #routes == V.
    Uses cost_mat (we'll pass time_mat).
    Returns routes as index lists (each starts/ends depot_idx).
    """
    routes = [[depot_idx, i, depot_idx] for i in cust_idx]

    def build_owner_map(routes_):
        owner = {}
        for ri, r in enumerate(routes_):
            for node in r[1:-1]:
                owner[node] = ri
        return owner

    savings = []
    for i in cust_idx:
        for j in cust_idx:
            if i == j:
                continue
            s = cost_mat[depot_idx, i] + cost_mat[depot_idx, j] - cost_mat[i, j]
            savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    def is_start(route, node):
        return len(route) >= 3 and route[1] == node

    def is_end(route, node):
        return len(route) >= 3 and route[-2] == node

    for s, i, j in savings:
        if len(routes) <= V:
            break

        owner = build_owner_map(routes)
        if i not in owner or j not in owner:
            continue

        ri = owner[i]
        rj = owner[j]
        if ri == rj:
            continue

        R_i = routes[ri]
        R_j = routes[rj]
        merged = None

        if is_end(R_i, i) and is_start(R_j, j):
            merged = R_i[:-1] + R_j[1:]
        elif is_end(R_j, j) and is_start(R_i, i):
            merged = R_j[:-1] + R_i[1:]
        elif is_start(R_i, i) and is_start(R_j, j):
            R_i_rev = [R_i[0]] + list(reversed(R_i[1:-1])) + [R_i[-1]]
            if is_end(R_i_rev, i) and is_start(R_j, j):
                merged = R_i_rev[:-1] + R_j[1:]
        elif is_end(R_i, i) and is_end(R_j, j):
            R_j_rev = [R_j[0]] + list(reversed(R_j[1:-1])) + [R_j[-1]]
            if is_end(R_i, i) and is_start(R_j_rev, j):
                merged = R_i[:-1] + R_j_rev[1:]

        if merged is None:
            continue

        keep = min(ri, rj)
        drop = max(ri, rj)
        routes[keep] = merged
        routes.pop(drop)

    if len(routes) > V:
        routes = sorted(routes, key=lambda r: route_travel_time(r))[:V]

    return routes

##################################################
# Build solutions: V=1 TSP baseline + CW-VRP for V=2..4
##################################################

V_list = [1, 2, 3, 4]
solutions = []
rows = []

for V in V_list:

    # ---- V=1 baseline: use already-solved TSP tour from Part-1 ----
    if V == 1:
        if "tour_idx" not in globals():
            raise NameError("tour_idx not found. Run the TSP baseline (Part-1) first to create tour_idx.")
        routes_idx = [tour_idx]
        met = compute_solution_metrics(routes_idx, S_max=S_max_per_vehicle)
        sol = {
            "V": 1,
            "method": "TSP (baseline, exact)",
            "routes_idx": routes_idx,
            "routes": [[idx_to_node[i] for i in r] for r in routes_idx],
            **met
        }

    # ---- V>=2: Clarke–Wright Savings heuristic ----
    else:
        routes_idx = clarke_wright_vrp(V, cost_mat=time_mat)
        met = compute_solution_metrics(routes_idx, S_max=S_max_per_vehicle)
        sol = {
            "V": V,
            "method": "Clarke–Wright Savings (heuristic)",
            "routes_idx": routes_idx,
            "routes": [[idx_to_node[i] for i in r] for r in routes_idx],
            **met
        }

    solutions.append(sol)
    rows.append({
        "V": sol["V"],
        "method": sol["method"],
        "total_travel_time_min": round(sol["total_travel_time_min"], 2),
        "total_distance_km": round(sol["total_distance_km"], 2),
        "max_route_total_time_min": round(sol["max_route_total_time_min"], 2),
        "feasible_460min": sol["feasible_460min"],
    })

summary_cw_df = pd.DataFrame(rows)
print("\n=== Routing Comparison (Selected Set fixed, V vehicles) ===")
print(summary_cw_df.to_string(index=False))

##################################################
# Visualization: plot routes for each V
##################################################
for sol in solutions:
    V = sol["V"]
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:,0], coords[:,1])

    for i in range(n):
        node = idx_to_node[i]
        if node == DEPOT_NODE:
            plt.text(coords[i,0], coords[i,1], f"DEPOT({node})", fontweight="bold")
        else:
            plt.text(coords[i,0], coords[i,1], str(node), fontsize=9)

    for r in sol["routes_idx"]:
        for a, b in zip(r[:-1], r[1:]):
            x0, y0 = coords[a]
            x1, y1 = coords[b]
            plt.plot([x0, x1], [y0, y1])

    title = f"Routing Solution (V vehicles = {V})\n{sol['method']}"
    plt.title(title)
    plt.xlabel("MDS dimension 1")
    plt.ylabel("MDS dimension 2")
    plt.grid(True)
    plt.show()

#################################################
# Route details (per vehicle)
##################################################
for sol in solutions:
    print(f"\n=== Route Details | V={sol['V']} | {sol['method']} ===")
    for v, rm in enumerate(sol["route_metrics"], start=1):
        print(f"Vehicle {v}:")
        print(f"  Customers: {rm['customers']}")
        print(f"  Travel time (min): {rm['travel_time_min']:.2f} | Service time (min): {rm['service_time_min']:.2f} | Total (min): {rm['route_total_time_min']:.2f}")
        print(f"  Distance (km): {rm['distance_km']:.2f}")


## REPORTING

cost_per_km = 0.35
S_max = S_max_per_vehicle  # 460

rows = []
for sol in solutions:
    route_totals = [rm["route_total_time_min"] for rm in sol["route_metrics"]]
    max_route_total = float(np.max(route_totals)) if route_totals else 0.0
    buffer_time = S_max - max_route_total

    total_distance = float(sol["total_distance_km"])
    travel_cost = total_distance * cost_per_km

    rows.append({
        "V": sol["V"],  # fleet size
        "method": sol["method"],
        "total_travel_time_min": round(float(sol["total_travel_time_min"]), 2),
        "total_distance_km": round(total_distance, 2),
        "travel_cost": round(travel_cost, 2),
        "max_route_total_time_min": round(max_route_total, 2),
        "buffer_time_min": round(buffer_time, 2),
        "feasible_460min": sol["feasible_460min"]
    })

kpi_df_simple = pd.DataFrame(rows).sort_values("V").reset_index(drop=True)

# İstersen distance_km veya feasible'ı bile çıkarabiliriz; ben şimdilik bıraktım.
print("\n=== Routing Evaluation KPIs (Simple) ===")
print(kpi_df_simple.to_string(index=False))

def knapsack_autoK(df, S_max_total, E_max_total,
                   id_col="request_id", value_col="value_score",
                   time_col="total_time_min", energy_col="kwh_needed",
                   verbose=False):
    ids = df[id_col].tolist()
    value  = dict(zip(df[id_col], df[value_col]))
    tproxy = dict(zip(df[id_col], df[time_col]))
    energy = dict(zip(df[id_col], df[energy_col]))

    m = gp.Model("knapsack_autoK")
    m.setParam("OutputFlag", 1 if verbose else 0)

    x = m.addVars(ids, vtype=GRB.BINARY, name="x")

    m.setObjective(gp.quicksum(value[i] * x[i] for i in ids), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(tproxy[i] * x[i] for i in ids) <= S_max_total, name="time_budget")
    m.addConstr(gp.quicksum(energy[i] * x[i] for i in ids) <= E_max_total, name="energy_budget")

    m.optimize()

    if m.SolCount == 0:
        return None

    sel = [i for i in ids if x[i].X > 0.5]
    total_value = sum(value[i] for i in sel)
    total_time  = sum(tproxy[i] for i in sel)
    total_kwh   = sum(energy[i] for i in sel)

    return {
        "selected_ids": set(sel),
        "selected_n": len(sel),
        "total_value_score": total_value,
        "total_time_proxy_min": total_time,
        "time_util": total_time / S_max_total,
        "total_energy_kwh": total_kwh,
        "energy_util": total_kwh / E_max_total
    }


##############################
# Run for multiple fleet sizes
###############################
V_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]  # proposal: vary K-vehicles (2–3) + baseline (1)

S_max_per_vehicle = 460
E_max_per_vehicle = 225

rows = []
sol_by_V = {}

for V in V_list:
    sol = knapsack_autoK(
        df,
        S_max_total = V * S_max_per_vehicle,
        E_max_total = V * E_max_per_vehicle,
        verbose=False
    )
    sol_by_V[V] = sol

    rows.append({
        "V": V,
        "selected_n (K*)": sol["selected_n"],
        "total_value_score": round(sol["total_value_score"], 2),
        "time_util": round(sol["time_util"], 4),
        "energy_util": round(sol["energy_util"], 4),
    })

knap_auto_df = pd.DataFrame(rows).sort_values("V").reset_index(drop=True)
print("\n=== Auto-K Knapsack vs Fleet Size ===")
print(knap_auto_df.to_string(index=False))


df_plot = knap_auto_df.copy()

df_plot["marginal_value_gain"] = df_plot["total_value_score"].diff()

fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(
    df_plot["V"],
    df_plot["total_value_score"],
    marker="o",
    linewidth=2,
    label="Total Value Score"
)
ax1.set_xlabel("Fleet Size (V)")
ax1.set_ylabel("Total Value Score")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(
    df_plot["V"],
    df_plot["marginal_value_gain"],
    marker="s",
    linestyle="--",
    linewidth=2,
    label="Marginal Value Gain (Δ Value)"
)
ax2.set_ylabel("Marginal Value Gain per Additional Vehicle")

plt.title("Auto-K Knapsack: Fleet Size vs Total Value and Marginal Value")

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

plt.tight_layout()
plt.show()
