# Blockhouse-Work-Trial-
# Smart Order Router - Cont & Kukanov Implementation

This project implements and backtests a Smart Order Router (SOR) based on the static cost model introduced by Cont & Kukanov in their paper "Optimal Order Placement in Limit Order Markets". The algorithm optimally splits orders across multiple trading venues to minimize execution costs.

## Implementation Approach

The implementation closely follows the provided pseudocode for the static Cont-Kukanov allocator. The main components include:

1. **Order Allocator**: The core algorithm that determines optimal order splitting across venues, exactly as specified in the pseudocode.

2. **Backtesting Engine**: Processes market data sequentially, executing orders according to the allocation strategy and tracking costs.

3. **Parameter Optimization**: A grid search that finds the optimal values for the three risk parameters: lambda_over, lambda_under, and theta_queue.

4. **Baseline Strategies**:
   - Best Ask: Simply routes the entire order to the venue with the best price
   - TWAP: Time-weighted average price, executing in 60-second buckets
   - VWAP: Volume-weighted average price, proportionally allocating based on displayed size

## Parameter Selection

The parameter search space covers a range of values for the three key risk parameters:

- **lambda_over**: Cost penalty per extra share bought
  - Values tested: [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]

- **lambda_under**: Cost penalty per unfilled share
  - Values tested: [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

- **theta_queue**: Queue-risk penalty factor
  - Values tested: [0.0, 0.005, 0.01, 0.05, 0.1]

The step size for allocation granularity is fixed at 100 shares, which balances optimization precision with computational efficiency.

## Results

The optimal parameters found through backtesting are directly output in the JSON results. The algorithm successfully compares the Cont & Kukanov approach against the three baseline strategies, measuring savings in basis points as required.

In our test run with the provided data:
- The Cont & Kukanov approach achieved modest but meaningful savings over the best-ask strategy
- More significant savings were seen compared to the TWAP strategy
- The performance was similar to VWAP, which is a more sophisticated baseline

## Suggested Improvement: Queue Position Modeling

The current implementation assumes immediate execution at displayed prices up to available size. A more realistic approach would incorporate queue position dynamics, which would significantly impact execution quality.

A proposed enhancement would:

1. **Estimate Queue Position**: Model the trader's position in each venue's order queue based on arrival time and market update patterns.

2. **Fill Probability Model**: Create a statistical model that estimates execution probability as a function of:
   - Queue position
   - Recent venue activity/volume
   - Time of day characteristics

3. **Time Decay Factor**: Add a time penalty to the cost function that increases as orders remain unfilled, capturing the increased risk of adverse price movements.

4. **Adaptive Re-routing**: Implement logic to periodically reassess and potentially re-route unfilled orders based on changing market conditions.

This improvement would create a more realistic simulation that accounts for the complex microstructure of limit order markets, where execution priority and queue dynamics significantly affect trading outcomes.
