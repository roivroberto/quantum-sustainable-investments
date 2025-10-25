import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
JSON_FILE = "big_dataset.json"
OUTPUT_FILE = "simulated_master_dataset.csv"
MIN_HISTORY_DAYS = 60
FINANCIAL_DRIFT_WEIGHT = 0.7
PEER_DRIFT_WEIGHT = 0.2
NOISE_WEIGHT = 0.1
NOISE_STD = 0.5
CONTROVERSY_THRESHOLD = 3
CONTROVERSY_SHOCK = -2.0
PEER_DRIFT_FACTOR = 0.001
MIN_DAYS_BEFORE_ANCHOR = 90

def load_json_data(filepath):
    """Load the JSON data file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def validate_company(ticker, company_data):
    """Check if company has required data for simulation."""
    # Check for ESG anchor
    sustainability = company_data.get("Sustainability", {})
    esg_scores = sustainability.get("esgScores", {})
    total_esg = esg_scores.get("totalEsg")
    
    if not total_esg:
        return False, None, None, None
    
    # Check for history
    history = company_data.get("History", {})
    if len(history) < MIN_HISTORY_DAYS:
        return False, None, None, None
    
    # Use the LAST DATE in history as the anchor date
    # The totalEsg is assumed to be the ESG score on that last date
    try:
        dates = sorted(history.keys())
        last_date_str = dates[-1]
        anchor_date = pd.Timestamp(last_date_str)
    except:
        return False, None, None, None
    
    return True, total_esg, anchor_date, esg_scores

def get_peer_esg_average(esg_scores):
    """Extract peer group average ESG score."""
    peer_performance = esg_scores.get("peerEsgScorePerformance", {})
    if isinstance(peer_performance, dict):
        return peer_performance.get("avg")
    return None

def check_high_controversy(esg_scores):
    """Check if company has high controversy score."""
    controversy_score = esg_scores.get("highestControversy", 0)
    return controversy_score >= CONTROVERSY_THRESHOLD

def prepare_history_dataframe(history_dict, ticker, name, peer_group):
    """Convert history dictionary to sorted DataFrame with returns."""
    dates = []
    closes = []
    
    for date_str, ohlcv in history_dict.items():
        dates.append(pd.Timestamp(date_str))
        closes.append(ohlcv.get("Close", np.nan))
    
    # Create DataFrame and sort by date
    df = pd.DataFrame({
        "date": dates,
        "close_price": closes,
        "ticker": ticker,
        "name": name,
        "peer_group": peer_group
    })
    df = df.sort_values("date").reset_index(drop=True)
    
    # Calculate daily returns
    df["daily_return"] = df["close_price"].pct_change()
    df["daily_return"] = df["daily_return"].fillna(0)
    
    # Calculate 30-day moving average of returns
    df["ma30_return"] = df["daily_return"].rolling(window=30, min_periods=1).mean()
    
    return df

def simulate_esg_backward(df, anchor_date, anchor_esg, peer_avg_esg, 
                          has_controversy, controversy_dates_range):
    """
    Simulate ESG scores backward from anchor date to earliest date.
    
    Formula: Score(Day X) = Score(Day X+1) - (Combined_Drift) + Noise
    """
    df = df.copy()
    df["simulated_esg_score"] = np.nan
    
    # Find anchor date in dataframe
    anchor_idx = (df["date"] - anchor_date).abs().idxmin()
    
    if anchor_idx == 0:
        # Anchor is the first date, nothing to simulate backward
        return df
    
    # Pick controversy date if applicable
    controversy_date = None
    if has_controversy:
        min_date_idx = 0
        max_date_idx = max(0, anchor_idx - (MIN_DAYS_BEFORE_ANCHOR / 1.0))  # Approximate
        
        if max_date_idx > min_date_idx + 5:  # Ensure we have room
            controversy_date_idx = int(np.random.uniform(min_date_idx, max_date_idx))
            controversy_date = df.loc[controversy_date_idx, "date"]
    
    # Backward simulation
    current_esg = anchor_esg
    
    for i in range(anchor_idx, -1, -1):
        current_date = df.loc[i, "date"]
        
        # Check if this is the controversy date
        if controversy_date and current_date == controversy_date:
            # Add shock (as we go backward, we add the shock magnitude)
            current_esg += abs(CONTROVERSY_SHOCK)
        
        # Calculate financial drift (30-day MA of returns)
        if i < len(df) - 1:
            ma30_return = df.loc[i + 1, "ma30_return"]
        else:
            ma30_return = 0
        
        # Positive returns in future mean company was investing in ESG -> lower ESG in past
        financial_drift = -ma30_return * 100 * FINANCIAL_DRIFT_WEIGHT
        
        # Peer reversion drift
        if peer_avg_esg:
            peer_drift = (current_esg - peer_avg_esg) * PEER_DRIFT_FACTOR * PEER_DRIFT_WEIGHT
        else:
            peer_drift = 0
        
        # Random noise
        noise = np.random.normal(0, NOISE_STD) * NOISE_WEIGHT
        
        # Combined drift
        combined_drift = financial_drift + peer_drift + noise
        
        # Update ESG score for current day
        new_esg = current_esg - combined_drift
        df.loc[i, "simulated_esg_score"] = new_esg
        current_esg = new_esg
    
    return df

def forward_fill_esg(df, anchor_date, anchor_esg):
    """
    Forward-fill ESG scores after anchor date with constant value.
    """
    df = df.copy()
    
    # Set anchor date and all dates after it to anchor_esg
    df.loc[df["date"] >= anchor_date, "simulated_esg_score"] = anchor_esg
    
    return df

def process_company(ticker, company_data):
    """Process a single company and return simulated data."""
    # Validate
    is_valid, total_esg, anchor_date, esg_scores = validate_company(ticker, company_data)
    if not is_valid:
        print(f"Skipping {ticker}: Missing required data")
        return None
    
    # Extract metadata
    name = company_data.get("Name", "")
    peer_group = company_data.get("Peer Group", "")
    history = company_data.get("History", {})
    
    # Prepare DataFrame
    df = prepare_history_dataframe(history, ticker, name, peer_group)
    
    # Get peer average ESG
    peer_avg_esg = get_peer_esg_average(esg_scores)
    
    # Check for controversy
    has_controversy = check_high_controversy(esg_scores)
    
    # Simulate backward
    df = simulate_esg_backward(df, anchor_date, total_esg, peer_avg_esg, 
                              has_controversy, (df["date"].min(), anchor_date))
    
    # Forward-fill after anchor date
    df = forward_fill_esg(df, anchor_date, total_esg)
    
    # Ensure no NaN values remain in simulated_esg_score
    df["simulated_esg_score"] = df["simulated_esg_score"].fillna(total_esg)
    
    # Select and rename columns for output
    df_output = df[[
        "date", "ticker", "name", "peer_group", 
        "close_price", "daily_return", "simulated_esg_score"
    ]].copy()
    
    df_output.columns = [
        "date", "ticker", "name", "peer_group",
        "close_price", "daily_return", "simulated_esg_score"
    ]
    
    return df_output

def main():
    """Main execution function."""
    print("Loading JSON data...")
    data = load_json_data(JSON_FILE)
    companies_raw = data.get("data", {})
    
    print(f"Found {len(companies_raw)} companies in JSON")
    
    # Process each company
    all_results = []
    valid_companies = 0
    
    for ticker, company_data in companies_raw.items():
        result = process_company(ticker, company_data)
        if result is not None:
            all_results.append(result)
            valid_companies += 1
            print(f"✓ Processed {ticker}")
        else:
            print(f"✗ Skipped {ticker}")
    
    print(f"\nSuccessfully processed {valid_companies} companies")
    
    if not all_results:
        print("ERROR: No companies were processed successfully")
        return
    
    # Combine all results
    master_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by date and ticker for readability
    master_df = master_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # Format columns
    master_df["date"] = master_df["date"].dt.strftime("%Y-%m-%d")
    master_df["close_price"] = master_df["close_price"].round(2)
    master_df["daily_return"] = master_df["daily_return"].round(6)
    master_df["simulated_esg_score"] = master_df["simulated_esg_score"].round(2)
    
    # Save to CSV
    master_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✓ Master dataset saved to {OUTPUT_FILE}")
    print(f"Total rows: {len(master_df)}")
    print(f"Date range: {master_df['date'].min()} to {master_df['date'].max()}")
    print(f"\nFirst few rows:")
    print(master_df.head(10))

if __name__ == "__main__":
    main()

