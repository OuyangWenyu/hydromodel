# Dahuofang (DHF) Model

The Dahuofang (DHF) model is a lumped conceptual hydrological model proposed by the Dahuofang Reservoir Administration in 1973. It consists of two main components: an 8-parameter infiltration excess runoff calculation model and an 8-parameter empirical unit hydrograph routing model with variable intensity and variable confluence velocity.

The model employs a dual-layer infiltration curve for loss calculation and uses a parabolic function to describe the upper layer water storage and dual-layer infiltration rate distribution. This is a lumped conceptual model specifically designed for flood forecasting applications.

## References

The implementation is primarily based on the Chinese textbook:
- **"水库防洪预报调度方法及应用" (Reservoir Flood Control Forecasting and Dispatching Methods and Applications)**, edited by Dalian University of Technology and the Office of State Flood Control and Drought Relief Headquarters, published by China Water & Power Press.

For English references, see:
- Li, X., Ye, L., Gu, X. et al. Development of A Distributed Modeling Framework Considering Spatiotemporally Varying Hydrological Processes for Sub-Daily Flood Forecasting in Semi-Humid and Semi-Arid Watersheds. *Water Resour Manage* 38, 3725–3754 (2024). https://doi.org/10.1007/s11269-024-03837-5

## Model Overview

The main entry point for the DHF model is the `dhf()` function, which provides a complete implementation of the Dahuofang hydrological model. This function integrates both runoff generation and routing components in a single, vectorized interface that can process multiple basins simultaneously.

Key features of the `dhf()` function:
- **Vectorized computation**: Processes all basins simultaneously using NumPy operations
- **Complete water balance**: Handles both runoff generation and channel routing
- **State management**: Supports warmup periods and state variable tracking  
- **Flexible configuration**: Configurable time intervals, basin parameters, and routing options
- **High performance**: Optimized for large-scale flood forecasting applications

## Model Components

### 1. Runoff Generation (产流计算)

The runoff generation component uses an 8-parameter infiltration excess model that:

- **Dual-layer infiltration curve**: Models water loss through a two-layer soil structure
- **Parabolic distribution**: Describes upper layer water storage capacity using parabolic functions
- **Surface and subsurface separation**: Distinguishes between surface runoff, interflow, and groundwater components
- **Evapotranspiration handling**: Accounts for both net precipitation and evaporation deficit conditions
- **Storage dynamics**: Updates soil moisture states for surface, subsurface, and deep layers

Key parameters include surface storage capacity (S0), subsurface storage capacity (U0), deep storage capacity (D0), and various coefficients controlling infiltration and flow generation.

### 2. Flow Routing (汇流计算)

The routing component implements an 8-parameter empirical unit hydrograph model featuring:

- **Variable intensity routing**: Adapts routing parameters based on antecedent conditions and current runoff intensity  
- **Variable confluence velocity**: Adjusts flow velocity according to flow magnitude and basin characteristics
- **Empirical unit hydrograph**: Uses trigonometric functions to describe the unit hydrograph shape
- **Surface and subsurface separation**: Routes surface flow and subsurface flow with different time constants
- **Channel parameters**: Incorporates main channel length, basin area, and routing coefficients

The routing process considers both immediate surface flow response and delayed subsurface contributions, providing realistic flood hydrograph simulation for Chinese watershed conditions.

## API Reference

::: hydromodel.models.dhf