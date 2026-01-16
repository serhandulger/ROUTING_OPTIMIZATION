# Routing Optimization & Knapsack Project

## Overview

This repository presents a **two-stage optimization framework** designed for daily operational planning of an **electric-vehicle (EV) mobile service provider** offering appointment-based roadside assistance / mobile charging within a single city.

The core business problem is deciding **which customer requests to serve** under limited operational capacity and **how to route service vehicles efficiently** for the selected requests. The project integrates **value-based selection** with **routing optimization**, providing a structured and data-driven decision-support tool.

---

## Business Problem Definition

Customers submit service requests **one day in advance**. Each request specifies:

* Required **energy (kWh)**
* Required **on-site service time (minutes)**
* Customer and commercial attributes contributing to a **business value score**

Because daily capacity is limited, the company faces two key questions:

1. **Selection Problem**
   *Which requests should be served tomorrow to maximize total business value under time and energy constraints?*

2. **Routing Problem**
   *Given the selected requests, how should they be visited in the field, and how does operational performance change with fleet size?*

The solution explicitly separates these decisions into **selection** and **routing**, allowing each to be optimized with the appropriate model.

---

## Solution Architecture (Two-Stage Optimization)

### Stage 1 — Selection: 0–1 Knapsack with Two Constraints

The first stage determines **which service requests to accept**.

* **Model type:** 0–1 Knapsack
* **Decision variable:**

  * `x_i = 1` if request *i* is selected, `0` otherwise
* **Objective:**

  * Maximize total **value_score** across selected requests
* **Constraints:**

  * Total energy usage ≤ daily kWh capacity
  * Total service time ≤ daily working-time capacity

Importantly, the model does **not** enforce a fixed number of requests. Instead, the optimal number of requests (**K*** ) emerges endogenously from capacity constraints.

#### Value Score Construction

Each request is assigned a single **value_score** capturing both short-term and long-term business impact:

* Customer segment (existing vs competitor-brand)
* Repurchase probability
* Acquisition probability
* Lifetime value (LTV)
* Brand / premium effects
* Operational difficulty indicators (distance from depot, traffic proxy)

This enables **value-driven prioritization**, ensuring that capacity is allocated to the most beneficial mix of requests rather than simply serving more jobs.


### Stage 2 — Routing: TSP and Multi-Vehicle VRP

Once the optimal request set is selected, routing feasibility and efficiency are evaluated.

#### 1) Single-Vehicle TSP (Baseline)

* **Purpose:** Validate whether the selected set is operationally feasible within a single day
* **Model:** Depot-based Traveling Salesman Problem
* **Objective:** Minimize total travel time

This step confirms that the knapsack solution is not only theoretically feasible, but also **route-feasible** when real inter-location travel times are used.

#### 2) Multi-Vehicle VRP (Clarke–Wright Savings Heuristic)

* **Purpose:** Analyze fleet-size trade-offs
* **Model:** K-vehicle Vehicle Routing Problem
* **Algorithm:** Clarke–Wright Savings heuristic

By varying the number of vehicles (V), the model quantifies:

* Total travel time and distance
* Workload of the busiest vehicle
* Operational buffer (slack vs daily time limit)

This makes the **cost vs robustness** trade-off explicit: fewer vehicles minimize cost, while additional vehicles increase reliability and reduce delay risk.

---

## Data Description

### Request-Level Dataset

* **Size:** 200 service requests
* **Key variables:**

  * `kwh_needed`
  * `service_time_min`
  * `x_km`, `y_km` (location coordinates)
  * `segment` (existing / competitor)
  * `repurchase_prob`, `acquisition_prob`, `ltv`, `brand_premium`

The dataset is synthetically generated to reflect a realistic operational environment by ChatGPT.

### Distance Matrix

* Fully connected depot–customer and customer–customer matrix
* Includes:

  * Distance (km)
  * Travel time (minutes)

Routing costs in TSP and VRP are computed using **actual travel times**, not proxies.

## Key Results & Insights

### Optimal Daily Operation (Single Vehicle)

* Maximum feasible workload under daily limits: **17 requests**
* Optimal operating point selected: **K = 13**
* Time constraint identified as the primary bottleneck

### Routing Validation

* Knapsack time proxy is intentionally conservative
* Actual TSP routing confirms feasibility with significant buffer
* ~32% of the operational day is spent traveling → routing efficiency is critical

### Fleet Size Trade-off

* Largest operational improvement occurs when moving from **1 → 2 vehicles**
* Additional vehicles provide diminishing marginal robustness
* Supports data-driven fleet deployment decisions

### Auto-K & Marginal Value Analysis

* As fleet size increases, both selected request count and total value increase
* Marginal value per additional vehicle declines
* Enables **effective fleet utilization** rather than fixed fleet expansion

## Business Value & Decision Support

This framework enables the firm to:

* Prioritize **high-value requests** under capacity constraints
* Validate selections at the **route level**, not just theoretically
* Quantify the **cost–robustness trade-off** of fleet size
* Decide daily **how many vehicles to deploy** based on marginal contribution

Overall, the project demonstrates how **combinatorial optimization** can directly support operational planning and strategic resource allocation.

## Algorithms Used

* 0–1 Knapsack (two constraints)
* Traveling Salesman Problem (TSP)
* Vehicle Routing Problem (VRP)
* Clarke–Wright Savings Heuristic

## Notes

This repository is intended for **academic and demonstrative purposes**. Data is synthetic and does not represent real customer information.
