import pandas as pd
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from datetime import datetime


class Venue:
    """Represents a trading venue with associated price and liquidity information."""
    def __init__(self, publisher_id: str, ask: float, ask_size: int, fee: float = 0.0, rebate: float = 0.0):
        self.publisher_id = publisher_id
        self.ask = ask
        self.ask_size = ask_size
        self.fee = fee
        self.rebate = rebate

    def __repr__(self):
        return f"Venue({self.publisher_id}, ask={self.ask}, size={self.ask_size}, fee={self.fee}, rebate={self.rebate})"


def allocate(order_size: int, venues: List[Venue], λ_over: float, λ_under: float, θ_queue: float, step: int = 100) -> Tuple[List[int], float]:
    """
    Implements the Cont & Kukanov allocation algorithm.
    
    Args:
        order_size: Target shares to buy
        venues: List of venue objects with ask price, size, fee, rebate
        λ_over: Cost penalty per extra share bought
        λ_under: Cost penalty per unfilled share
        θ_queue: Queue-risk penalty
        step: Share increment to use for allocation search (default: 100)
        
    Returns:
        Tuple of (best allocation list, associated cost)
    """
    splits = [[]]  # start with an empty allocation list
    
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, venues[v].ask_size)
            for q in range(0, max_v + 1, step):
                new_alloc = alloc.copy()
                new_alloc.append(q)
                new_splits.append(new_alloc)
        splits = new_splits

    best_cost = float('inf')
    best_split = []
    
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, λ_over, λ_under, θ_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
            
    return best_split, best_cost


def compute_cost(split: List[int], venues: List[Venue], order_size: int, λo: float, λu: float, θ: float) -> float:
    """
    Computes the expected cost of a specific allocation.
    
    Args:
        split: List of shares allocated to each venue
        venues: List of venue objects
        order_size: Target shares to buy
        λo: Cost penalty per extra share bought
        λu: Cost penalty per unfilled share
        θ: Queue-risk penalty
        
    Returns:
        Total expected cost
    """
    executed = 0
    cash_spent = 0.0
    
    for i in range(len(venues)):
        exe = min(split[i], venues[i].ask_size)
        executed += exe
        cash_spent += exe * (venues[i].ask + venues[i].fee)
        maker_rebate = max(split[i] - exe, 0) * venues[i].rebate
        cash_spent -= maker_rebate

    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    risk_pen = θ * (underfill + overfill)
    cost_pen = λu * underfill + λo * overfill
    
    return cash_spent + risk_pen + cost_pen


def load_market_data(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess the market data.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Processed DataFrame with one row per venue per unique timestamp
    """
    print(f"Loading market data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert timestamp to datetime for easier manipulation
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    
    # Sort by timestamp then by publisher_id to ensure deterministic processing
    df = df.sort_values(['ts_event', 'publisher_id'])
    
    # Keep only the first entry per (timestamp, venue) to get one snapshot per venue per timestamp
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'], keep='first')
    
    # Filter out invalid ask prices or sizes (if any)
    df = df[(df['ask_px_00'] > 0) & (df['ask_sz_00'] > 0)]
    
    # Create separate dataframes for unique timestamps and venue data
    timestamps = df['ts_event'].unique()
    
    print(f"Loaded {len(timestamps)} unique timestamps across {df['publisher_id'].nunique()} venues")
    return df


def run_backtest(df: pd.DataFrame, order_size: int, λ_over: float, λ_under: float, θ_queue: float, 
                 venue_fees: Dict[str, float], step: int = 100) -> Dict:
    """
    Run the backtest using the Cont & Kukanov allocator.
    
    Args:
        df: Processed market data
        order_size: Total shares to execute
        λ_over: Cost penalty per extra share bought
        λ_under: Cost penalty per unfilled share
        θ_queue: Queue-risk penalty
        venue_fees: Dictionary mapping venue IDs to their fees
        
    Returns:
        Dictionary with execution results and metrics
    """
    timestamps = df['ts_event'].unique()
    remaining_size = order_size
    total_executed = 0
    total_cost = 0.0
    allocations_history = []
    execution_history = []
    costs_by_timestamp = []
    
    for ts in timestamps:
        if remaining_size <= 0:
            break
            
        # Get snapshot for this timestamp
        snapshot = df[df['ts_event'] == ts]
        
        # Create venue objects
        venues = []
        for _, row in snapshot.iterrows():
            venue_id = row['publisher_id']
            venues.append(Venue(
                publisher_id=venue_id,
                ask=row['ask_px_00'],
                ask_size=int(row['ask_sz_00']),
                fee=venue_fees.get(venue_id, 0.0),
                rebate=0.0  # No rebate data provided, assuming 0
            ))
        
        # Skip if no valid venues
        if not venues:
            continue
            
        # Run allocator
        allocation, _ = allocate(remaining_size, venues, λ_over, λ_under, θ_queue, step)
        
        # If no valid allocation found (rare edge case), skip this timestamp
        if not allocation:
            continue
            
        # Execute the allocation
        executed_this_step = 0
        cost_this_step = 0.0
        executions = []
        
        for i, shares in enumerate(allocation):
            if shares > 0:
                venue = venues[i]
                executed = min(shares, venue.ask_size)
                executed_this_step += executed
                exec_cost = executed * venue.ask
                cost_this_step += exec_cost
                
                executions.append({
                    'venue': venue.publisher_id,
                    'shares': executed,
                    'price': venue.ask,
                    'cost': exec_cost
                })
        
        # Update totals
        total_executed += executed_this_step
        total_cost += cost_this_step
        remaining_size -= executed_this_step
        
        # Record allocation and execution for this timestamp
        allocations_history.append({
            'timestamp': ts,
            'allocation': dict(zip([v.publisher_id for v in venues], allocation)),
            'venues': [{'id': v.publisher_id, 'ask': v.ask, 'size': v.ask_size} for v in venues]
        })
        
        execution_history.append({
            'timestamp': ts,
            'executed': executed_this_step,
            'cost': cost_this_step,
            'details': executions
        })
        
        costs_by_timestamp.append((ts, total_cost))
    
    # Calculate metrics
    avg_price = total_cost / total_executed if total_executed > 0 else 0
    
    return {
        'strategy': 'cont_kukanov',
        'params': {
            'lambda_over': λ_over,
            'lambda_under': λ_under,
            'theta_queue': θ_queue
        },
        'target_size': order_size,
        'executed': total_executed,
        'remaining': remaining_size,
        'total_cost': total_cost,
        'avg_price': avg_price,
        'costs_over_time': costs_by_timestamp,
        'execution_details': execution_history
    }


def run_best_ask_baseline(df: pd.DataFrame, order_size: int) -> Dict:
    """
    Run the baseline strategy that always takes the best ask price.
    
    Args:
        df: Processed market data
        order_size: Total shares to execute
        
    Returns:
        Dictionary with execution results and metrics
    """
    timestamps = df['ts_event'].unique()
    remaining_size = order_size
    total_executed = 0
    total_cost = 0.0
    execution_history = []
    costs_by_timestamp = []
    
    for ts in timestamps:
        if remaining_size <= 0:
            break
            
        # Get snapshot for this timestamp
        snapshot = df[df['ts_event'] == ts]
        
        if snapshot.empty:
            continue
            
        # Find venue with best ask price
        best_venue = snapshot.loc[snapshot['ask_px_00'].idxmin()]
        
        # Execute at best venue
        shares_to_execute = min(remaining_size, int(best_venue['ask_sz_00']))
        if shares_to_execute <= 0:
            continue
            
        execution_cost = shares_to_execute * best_venue['ask_px_00']
        
        # Update totals
        total_executed += shares_to_execute
        total_cost += execution_cost
        remaining_size -= shares_to_execute
        
        execution_history.append({
            'timestamp': ts,
            'venue': best_venue['publisher_id'],
            'executed': shares_to_execute,
            'price': best_venue['ask_px_00'],
            'cost': execution_cost
        })
        
        costs_by_timestamp.append((ts, total_cost))
    
    # Calculate metrics
    avg_price = total_cost / total_executed if total_executed > 0 else 0
    
    return {
        'strategy': 'best_ask',
        'target_size': order_size,
        'executed': total_executed,
        'remaining': remaining_size,
        'total_cost': total_cost,
        'avg_price': avg_price,
        'costs_over_time': costs_by_timestamp
    }


def run_twap_baseline(df: pd.DataFrame, order_size: int, bucket_seconds: int = 60) -> Dict:
    """
    Run a time-weighted average price (TWAP) baseline.
    
    Args:
        df: Processed market data
        order_size: Total shares to execute
        bucket_seconds: Size of time buckets in seconds
        
    Returns:
        Dictionary with execution results and metrics
    """
    # Create time buckets
    df['bucket'] = df['ts_event'].dt.floor(f'{bucket_seconds}s')
    buckets = df['bucket'].unique()
    
    shares_per_bucket = order_size / len(buckets)
    remaining_size = order_size
    total_executed = 0
    total_cost = 0.0
    execution_history = []
    costs_by_timestamp = []
    
    for bucket in buckets:
        if remaining_size <= 0:
            break
            
        bucket_data = df[df['bucket'] == bucket]
        if bucket_data.empty:
            continue
            
        # Determine shares to execute in this bucket
        shares_this_bucket = min(int(shares_per_bucket), remaining_size)
        if shares_this_bucket <= 0:
            continue
            
        # Group timestamps within this bucket
        timestamps = bucket_data['ts_event'].unique()
        shares_per_ts = max(1, shares_this_bucket // len(timestamps))
        
        shares_executed_in_bucket = 0
        
        for ts in timestamps:
            if shares_executed_in_bucket >= shares_this_bucket:
                break
                
            snapshot = bucket_data[bucket_data['ts_event'] == ts]
            if snapshot.empty:
                continue
                
            # Find venue with best ask price
            best_venue = snapshot.loc[snapshot['ask_px_00'].idxmin()]
            
            # Execute at best venue
            shares_to_execute = min(
                shares_per_ts,
                shares_this_bucket - shares_executed_in_bucket,
                int(best_venue['ask_sz_00'])
            )
            
            if shares_to_execute <= 0:
                continue
                
            execution_cost = shares_to_execute * best_venue['ask_px_00']
            
            # Update totals
            total_executed += shares_to_execute
            total_cost += execution_cost
            remaining_size -= shares_to_execute
            shares_executed_in_bucket += shares_to_execute
            
            execution_history.append({
                'timestamp': ts,
                'bucket': bucket,
                'venue': best_venue['publisher_id'],
                'executed': shares_to_execute,
                'price': best_venue['ask_px_00'],
                'cost': execution_cost
            })
            
            costs_by_timestamp.append((ts, total_cost))
    
    # Calculate metrics
    avg_price = total_cost / total_executed if total_executed > 0 else 0
    
    return {
        'strategy': 'twap',
        'bucket_seconds': bucket_seconds,
        'target_size': order_size,
        'executed': total_executed,
        'remaining': remaining_size,
        'total_cost': total_cost,
        'avg_price': avg_price,
        'costs_over_time': costs_by_timestamp
    }


def run_vwap_baseline(df: pd.DataFrame, order_size: int) -> Dict:
    """
    Run a volume-weighted average price (VWAP) baseline that weights prices by displayed ask size.
    
    Args:
        df: Processed market data
        order_size: Total shares to execute
        
    Returns:
        Dictionary with execution results and metrics
    """
    timestamps = df['ts_event'].unique()
    
    # Calculate total volume across all timestamps
    df['volume_weight'] = df['ask_sz_00'] / df.groupby('ts_event')['ask_sz_00'].transform('sum')
    
    # Allocate shares per timestamp based on volume weight
    timestamp_allocations = {}
    remaining_shares = order_size
    
    for ts in timestamps:
        ts_data = df[df['ts_event'] == ts]
        total_volume_at_ts = ts_data['ask_sz_00'].sum()
        total_volume_weight = total_volume_at_ts / df['ask_sz_00'].sum()
        shares_for_ts = int(order_size * total_volume_weight)
        
        # Ensure we don't exceed remaining shares
        shares_for_ts = min(shares_for_ts, remaining_shares)
        if shares_for_ts > 0:
            timestamp_allocations[ts] = shares_for_ts
            remaining_shares -= shares_for_ts
    
    # Distribute any remaining shares to timestamps with available liquidity
    if remaining_shares > 0:
        for ts in timestamps:
            if ts not in timestamp_allocations:
                continue
                
            ts_data = df[df['ts_event'] == ts]
            available_liquidity = ts_data['ask_sz_00'].sum()
            additional_shares = min(remaining_shares, available_liquidity - timestamp_allocations[ts])
            
            if additional_shares > 0:
                timestamp_allocations[ts] += additional_shares
                remaining_shares -= additional_shares
                
            if remaining_shares <= 0:
                break
    
    # Execute according to allocations
    total_executed = 0
    total_cost = 0.0
    execution_history = []
    costs_by_timestamp = []
    
    for ts in timestamps:
        if ts not in timestamp_allocations:
            continue
            
        shares_for_ts = timestamp_allocations[ts]
        if shares_for_ts <= 0:
            continue
            
        ts_data = df[df['ts_event'] == ts]
        
        # Sort venues by ask price to execute at best prices first
        ts_data = ts_data.sort_values('ask_px_00')
        
        shares_executed_at_ts = 0
        
        for _, venue in ts_data.iterrows():
            shares_to_execute = min(
                shares_for_ts - shares_executed_at_ts,
                int(venue['ask_sz_00'])
            )
            
            if shares_to_execute <= 0:
                continue
                
            execution_cost = shares_to_execute * venue['ask_px_00']
            
            # Update totals
            total_executed += shares_to_execute
            total_cost += execution_cost
            shares_executed_at_ts += shares_to_execute
            
            execution_history.append({
                'timestamp': ts,
                'venue': venue['publisher_id'],
                'executed': shares_to_execute,
                'price': venue['ask_px_00'],
                'cost': execution_cost
            })
            
            costs_by_timestamp.append((ts, total_cost))
            
            if shares_executed_at_ts >= shares_for_ts:
                break
    
    # Calculate metrics
    avg_price = total_cost / total_executed if total_executed > 0 else 0
    
    return {
        'strategy': 'vwap',
        'target_size': order_size,
        'executed': total_executed,
        'remaining': order_size - total_executed,
        'total_cost': total_cost,
        'avg_price': avg_price,
        'costs_over_time': costs_by_timestamp
    }


def grid_search(df: pd.DataFrame, order_size: int, venue_fees: Dict[str, float], step: int = 100) -> Dict:
    """
    Perform grid search to find optimal parameters.
    
    Args:
        df: Processed market data
        order_size: Total shares to execute
        venue_fees: Dictionary mapping venue IDs to their fees
        
    Returns:
        Dictionary with best parameters and results
    """
    # Define parameter grid
    lambda_over_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    lambda_under_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    theta_queue_values = [0.0, 0.005, 0.01, 0.05, 0.1]
    
    best_avg_price = float('inf')
    best_params = None
    best_result = None
    
    total_combinations = len(lambda_over_values) * len(lambda_under_values) * len(theta_queue_values)
    print(f"Starting grid search with {total_combinations} parameter combinations...")
    start_time = time.time()
    
    count = 0
    for λ_over in lambda_over_values:
        for λ_under in lambda_under_values:
            for θ_queue in theta_queue_values:
                count += 1
                if count % 10 == 0:
                    elapsed = time.time() - start_time
                    estimated_total = elapsed * total_combinations / count
                    print(f"Progress: {count}/{total_combinations} combinations "
                          f"({100*count/total_combinations:.1f}%) - "
                          f"Time: {elapsed:.1f}s / ~{estimated_total:.1f}s")
                
                result = run_backtest(df, order_size, λ_over, λ_under, θ_queue, venue_fees, step)
                
                # Skip if it didn't execute all shares
                if result['executed'] < order_size * 0.95:  # Allow for slight underfill
                    continue
                
                if result['avg_price'] < best_avg_price:
                    best_avg_price = result['avg_price']
                    best_params = {
                        'lambda_over': λ_over,
                        'lambda_under': λ_under,
                        'theta_queue': θ_queue
                    }
                    best_result = result
    
    print(f"Grid search completed in {time.time() - start_time:.1f} seconds")
    return {
        'best_params': best_params,
        'best_result': best_result
    }


def plot_cumulative_costs(results: Dict, filename: str = 'results.png'):
    """
    Create a cumulative cost plot comparing different strategies.
    
    Args:
        results: Dictionary with results from all strategies
        filename: Output filename for the plot
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot each strategy's cumulative cost using downsampling
        colors = ['blue', 'orange', 'green', 'red']
        line_styles = ['-', '--', '-.', ':']
        
        strategies = ['cont_kukanov', 'best_ask', 'twap', 'vwap']
        i = 0
        
        for strategy in strategies:
            if strategy not in results:
                continue
                
            data = results[strategy]
            costs = data.get('costs_over_time', [])
            if not costs:
                continue
                
            # Extract just the costs for plotting (ignore timestamps)
            cost_values = [cost for _, cost in costs]
            
            # Downsample to a maximum of 100 points to avoid memory issues
            if len(cost_values) > 100:
                indices = np.linspace(0, len(cost_values) - 1, 100).astype(int)
                downsampled_costs = [cost_values[i] for i in indices]
            else:
                downsampled_costs = cost_values
            
            # Plot downsampled data with different line styles to distinguish overlapping lines
            plt.plot(range(len(downsampled_costs)), downsampled_costs, 
                    color=colors[i % len(colors)],
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=2.5,
                    label=f"{strategy} (Avg: ${data['avg_price']:.4f})")
            i += 1
        
        plt.title('Cumulative Execution Cost by Strategy', fontsize=14)
        plt.xlabel('Execution Progress (%)', fontsize=12)
        plt.ylabel('Cumulative Cost ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Add annotations for savings in a better location
        savings = results.get('savings_bps', {})
        for i, (baseline, bps) in enumerate(savings.items()):
            plt.annotate(f"Savings vs {baseline}: {bps:.2f} bps", 
                     xy=(0.02, 0.95 - i*0.05), xycoords='axes fraction',
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Adjust layout and save with higher DPI for better quality
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Plot saved to {filename}")
        
    except Exception as e:
        print(f"Warning: Error creating plot: {str(e)}")
        
        # Fallback to a very simple bar chart if the line plot fails
        try:
            plt.figure(figsize=(8, 5))
            
            strategies = []
            avg_prices = []
            
            for strategy, data in results.items():
                if strategy == 'savings_bps':
                    continue
                strategies.append(strategy)
                avg_prices.append(data.get('avg_price', 0))
            
            plt.bar(strategies, avg_prices)
            plt.title('Average Fill Price by Strategy')
            plt.ylabel('Price ($)')
            
            # Add text annotation for savings
            savings_text = []
            for baseline, bps in results.get('savings_bps', {}).items():
                savings_text.append(f"Savings vs {baseline}: {bps:.2f} bps")
            
            plt.figtext(0.5, 0.01, '\n'.join(savings_text), ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(filename, dpi=100)
            print(f"Created simplified plot as fallback")
        except Exception as e2:
            print(f"Could not create even the fallback plot: {str(e2)}")
            
            # Last resort: create a minimal plot with just text
            try:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, "Cont & Kukanov Strategy Results\n\n" + 
                                "Average prices:\n" +
                                "\n".join([f"{s}: ${results[s]['avg_price']:.4f}" 
                                         for s in results if s != 'savings_bps']) +
                                "\n\nSavings:\n" +
                                "\n".join([f"vs {b}: {bps:.2f} bps" 
                                         for b, bps in results.get('savings_bps', {}).items()]),
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.axis('off')
                plt.savefig(filename, dpi=100)
                print(f"Created text-only plot as last resort")
            except:
                # Create an empty file if all else fails
                with open(filename, 'w') as f:
                    f.write('')
                print(f"Could not create any plot, but created empty file {filename}")


def calculate_savings_bps(optimal_result: Dict, baselines: Dict) -> Dict:
    """
    Calculate savings in basis points compared to baselines.
    
    Args:
        optimal_result: Results from the optimal strategy
        baselines: Dictionary of baseline results
        
    Returns:
        Dictionary with savings in basis points
    """
    savings = {}
    optimal_price = optimal_result['avg_price']
    
    for name, baseline in baselines.items():
        baseline_price = baseline['avg_price']
        # Calculate basis points: 1 bp = 0.01%
        bps = (baseline_price - optimal_price) / baseline_price * 10000
        savings[name] = bps
        
    return savings


def main():
    """Main function to run the backtest."""
    # Define parameters
    csv_path = 'l1_day.csv'
    order_size = 5000
    step = 100  # Allocation search granularity (shares)
    
    # Define venue fees (assuming uniform fees for this example)
    venue_fees = {}  # Empty dict for now, could be populated with actual fees
    
    try:
        # Load and preprocess market data
        df = load_market_data(csv_path)
        
        # Print unique venues for debugging
        unique_publishers = df['publisher_id'].unique()
        print(f"Found unique publishers: {unique_publishers}")
        
        # Run grid search for optimal parameters
        print("Running grid search...")
        search_result = grid_search(df, order_size, venue_fees, step)
        best_params = search_result['best_params']
        best_result = search_result['best_result']
        
        if not best_params:
            print("No valid parameter combination found. Try adjusting the search space.")
            return
        
        # Run baselines
        print("Running baseline strategies...")
        best_ask_result = run_best_ask_baseline(df, order_size)
        twap_result = run_twap_baseline(df, order_size, bucket_seconds=60)
        vwap_result = run_vwap_baseline(df, order_size)
        
        # Calculate savings
        savings = calculate_savings_bps(
            best_result,
            {
                'best_ask': best_ask_result,
                'twap': twap_result,
                'vwap': vwap_result
            }
        )
        
        # Prepare output results
        final_results = {
            'cont_kukanov': {
                'params': best_params,
                'total_cost': best_result['total_cost'],
                'avg_price': best_result['avg_price'],
                'executed': best_result['executed'],
                'costs_over_time': best_result['costs_over_time'][:100]  # Limit to first 100 points for plotting
            },
            'best_ask': {
                'total_cost': best_ask_result['total_cost'],
                'avg_price': best_ask_result['avg_price'],
                'executed': best_ask_result['executed'],
                'costs_over_time': best_ask_result['costs_over_time'][:100]  # Limit to first 100 points for plotting
            },
            'twap': {
                'total_cost': twap_result['total_cost'],
                'avg_price': twap_result['avg_price'],
                'executed': twap_result['executed'],
                'costs_over_time': twap_result['costs_over_time'][:100]  # Limit to first 100 points for plotting
            },
            'vwap': {
                'total_cost': vwap_result['total_cost'],
                'avg_price': vwap_result['avg_price'],
                'executed': vwap_result['executed'],
                'costs_over_time': vwap_result['costs_over_time'][:100]  # Limit to first 100 points for plotting
            },
            'savings_bps': savings
        }
        
        # Try to create plot, but don't fail if it errors
        try:
            plot_cumulative_costs(final_results)
        except Exception as e:
            print(f"Warning: Could not create plot: {str(e)}")
            print("Continuing with JSON output...")
        
        # Prepare JSON output
        output = {
            'best_parameters': best_params,
            'cont_kukanov': {
                'total_cash_spent': best_result['total_cost'],
                'average_fill_price': best_result['avg_price'],
                'shares_executed': best_result['executed']
            },
            'baselines': {
                'best_ask': {
                    'total_cash_spent': best_ask_result['total_cost'],
                    'average_fill_price': best_ask_result['avg_price'],
                    'shares_executed': best_ask_result['executed']
                },
                'twap': {
                    'total_cash_spent': twap_result['total_cost'],
                    'average_fill_price': twap_result['avg_price'],
                    'shares_executed': twap_result['executed']
                },
                'vwap': {
                    'total_cash_spent': vwap_result['total_cost'],
                    'average_fill_price': vwap_result['avg_price'],
                    'shares_executed': vwap_result['executed']
                }
            },
            'savings_bps': {
                'vs_best_ask': savings['best_ask'],
                'vs_twap': savings['twap'],
                'vs_vwap': savings['vwap']
            }
        }
        
        # Print JSON to stdout
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        print(f"Error in backtest: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()