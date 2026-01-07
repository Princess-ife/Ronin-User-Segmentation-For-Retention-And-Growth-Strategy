import requests
import pandas as pd
import time
import json

# Your Dune API credentials
API_KEY = "AjzHSSu2djGOja7cKnNH1J6opLZKVw2J"
QUERY_ID = "6451847"

# Dune API endpoint
BASE_URL = "https://api.dune.com/api/v1/query"

def fetch_dune_data(query_id, api_key):
    """
    Fetch query results from Dune API
    """
    try:
        # Step 1: Execute the query
        print(f"Executing query {query_id}...")
        execute_url = f"{BASE_URL}/{query_id}/execute"
        
        headers = {
            "X-DUNE-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        
        # Execute the query
        response = requests.post(execute_url, headers=headers, timeout=10)
        print(f"Execute response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error executing query: {response.text}")
            return None
        
        execution_id = response.json()["execution_id"]
        print(f"Query executing with ID: {execution_id}")
        
        # Step 2: Wait for query to complete
        print("Waiting for query to complete...")
        status_url = f"{BASE_URL}/{query_id}/executions/{execution_id}"
        
        max_wait = 60  # Wait max 60 seconds
        elapsed = 0
        
        while elapsed < max_wait:
            status_response = requests.get(status_url, headers=headers, timeout=10)
            response_json = status_response.json()
            status = response_json.get("state", "UNKNOWN")
            
            print(f"Status: {status}")
            
            if status == "QUERY_STATE_COMPLETED":
                print("Query completed!")
                break
            elif status == "QUERY_STATE_FAILED":
                print(f"Query failed: {response_json}")
                return None
            else:
                time.sleep(2)
                elapsed += 2
        
        if elapsed >= max_wait:
            print("Query took too long to complete")
            return None
        
        # Step 3: Get the results
        print("Fetching results...")
        results_response = requests.get(status_url, headers=headers, timeout=10)
        results_data = results_response.json()
        
        if "result" in results_data and "rows" in results_data["result"]:
            results = results_data["result"]["rows"]
            print(f"Got {len(results)} rows!")
            return results
        else:
            print(f"Unexpected response format: {results_data}")
            return None
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Fetch the data
print("Starting data fetch...")
data = fetch_dune_data(QUERY_ID, API_KEY)

if data:
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv("dune_wallet_data.csv", index=False)
    print("✓ Data saved to 'dune_wallet_data.csv'")
    
    # Show first few rows
    print("\nFirst few rows of your data:")
    print(df.head())
    
    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    print("✗ Failed to fetch data")