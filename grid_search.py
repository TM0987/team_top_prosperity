# grid_search.py
import itertools
import subprocess
import json
import jsonpickle # Use the same serializer as in arb_trader
import copy
import os
from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    # Add new products
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBE = "DJEMBES"
# It's better to define the default structure here or load it robustly
# For simplicity, let's redefine it (ensure it matches arb_trader.py)
DEFAULT_PARAMS = {
    # Parameters from v1.py
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    # Parameters for basket1 strategy
    "basket1_strategy": {
        "zscore_threshold": 0,
        "min_profit_margin": 2, # Specific margin for basket 1 (if needed)
        "max_position_pct": 1, # Specific position % for basket 1
        "history_length": 1, # Specific history length for basket 1
    },
    # Parameters for basket2 strategy
    "basket2_strategy": {
        "zscore_threshold": 2,
        "min_profit_margin": 2, # Specific margin for basket 2
        "max_position_pct": 1, # Specific position % for basket 2
        "history_length": 75, # Specific history length for basket 2
    }
}

# Define the parameter file name (must match arb_trader.py)
PARAMS_FILENAME = "params.json"

def run_simulation(params, simulation_command_list):
    """
    Runs the IMC simulation with the given parameters via a temporary file
    and returns the PnL.
    """
    params_json = jsonpickle.encode(params, indent=4) # Use indent for readability

    try:
        # Write the current parameters to the file
        with open(PARAMS_FILENAME, 'w') as f:
            f.write(params_json)
        print(f"Wrote parameters to {PARAMS_FILENAME} for this run.")

        # Execute the simulation runner script/command
        print(f"Running command: {' '.join(simulation_command_list)}")
        result = subprocess.run(
            simulation_command_list, # Pass command as a list for robustness
            shell=False, # Set shell=False when passing a list
            capture_output=True,
            text=True,
            check=True, # Raise exception on non-zero exit code
            # cwd=os.path.dirname(simulation_command_list[0]) # Ensure running from exe's dir if needed
            # Or adjust cwd if params.json needs to be relative to arb_trader.py
        )

        # --- Parse the PnL from the simulation output ---
        # This depends heavily on what prosperity3bt.exe prints
        # Example: Assume PnL is printed like "Final PnL: 1234.5"
        pnl = 0.0
        print("--- Simulation Output ---")
        output_lines = result.stdout.splitlines()
        for line in output_lines:
            print(line) # Print output for debugging PnL parsing
            # **** ADJUST THIS PARSING LOGIC ****
            if "Total profit:" in line: # MODIFY this condition based on actual output
                try:
                    pnl = float(line.split(":")[-1].strip())
                    print(f"Parsed PnL: {pnl}")
                    break
                except ValueError:
                    print(f"Warning: Could not parse PnL from line: {line}")
            # Add other potential PnL output formats here
            # elif "profit_and_loss" in line.lower(): ...

        if pnl == 0.0:
             print("Warning: PnL was not parsed from output.")

        print("--- End Simulation Output ---")
        print(f"Simulation finished. Reported PnL: {pnl}")
        return pnl

    except subprocess.CalledProcessError as e:
        print(f"Error running simulation command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print(f"Output:\n{e.stdout}")
        print(f"Error:\n{e.stderr}")
        return -float('inf') # Indicate failure
    except Exception as e:
        print(f"General error during simulation run: {e}")
        return -float('inf')
    # finally:
    #     # Clean up the parameter file
    #     if os.path.exists(PARAMS_FILENAME):
    #         try:
    #             os.remove(PARAMS_FILENAME)
    #             print(f"Removed temporary file: {PARAMS_FILENAME}")
    #         except Exception as e:
    #             print(f"Warning: Could not remove temporary file {PARAMS_FILENAME}: {e}")

# --- Define Parameter Grids ---
# (Keep your desired param_grid definition)
param_grid = {
    ('basket1_strategy', 'zscore_threshold'): [0.25, 0.5, 0.75, 1],
    ('basket1_strategy', 'min_profit_margin'): [1, 2],
    ('basket1_strategy', 'max_position_pct'): [1, 2],
    ('basket1_strategy', "history_length"): [25, 50, 75],
}



# --- Simulation Setup ---
# Define the command components as a list
# Adjust path to prosperity3bt.exe if it's not in the same directory or in PATH
SIMULATION_COMMAND_LIST = [r'prosperity3bt.exe', r'arb_trader.py', '2']
# Ensure the path to arb_trader.py is correct relative to where grid_search.py runs

# --- Grid Search Execution ---
results = []
best_pnl = -float('inf')
best_params_combination = None # Store the winning combination dict

# Generate all combinations of parameter values
keys, values = zip(*param_grid.items())
parameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Starting grid search with {len(parameter_combinations)} combinations...")

for i, param_combination in enumerate(parameter_combinations):
    print(f"\n--- Combination {i+1}/{len(parameter_combinations)} ---")
    current_params = copy.deepcopy(DEFAULT_PARAMS) # Start with defaults

    # Apply the current combination to the params dictionary
    print("Applying parameters:")
    for (key1, key2), value in param_combination.items():
        if key1 in current_params and isinstance(current_params[key1], dict) and key2 in current_params[key1]:
             current_params[key1][key2] = value
             print(f"  Setting {key1}.{key2} = {value}")
        # Handle cases where the top-level key might not have a dict (though unlikely based on your PARAMS)
        elif key1 in current_params and key2 is None: # If optimizing a top-level param directly
             current_params[key1] = value
             print(f"  Setting {key1} = {value}")
        else:
             print(f"  Warning: Parameter key ({key1}, {key2}) not found in defaults structure.")


    # Run the simulation with these parameters
    pnl = run_simulation(current_params, SIMULATION_COMMAND_LIST)
    results.append({'params': param_combination, 'pnl': pnl})

    # Update best result
    if pnl > best_pnl:
        best_pnl = pnl
        best_params_combination = param_combination
        print(f"*** New best PnL: {best_pnl} ***")

# --- Print Results ---
print("\n--- Grid Search Complete ---")
print(f"Best PnL found: {best_pnl}")
print("Best Parameter Combination:")
if best_params_combination:
    # Regenerate the full best parameter set for clarity if needed
    # best_full_params = copy.deepcopy(DEFAULT_PARAMS)
    # for (key1, key2), value in best_params_combination.items():
    #     if key1 in best_full_params and key2 in best_full_params[key1]:
    #          best_full_params[key1][key2] = value
    # print(json.dumps(best_full_params, indent=4))
    # Or just print the combination that was varied:
    for (key1, key2), value in best_params_combination.items():
        print(f"  {key1}.{key2}: {value}")
else:
    print("  No successful simulations found or no PnL > -inf.")

# Optionally save all results
# with open("grid_search_results.json", "w") as f:
#    # Need to handle jsonpickle objects if params contain them, otherwise use standard json
#    json.dump(results, f, indent=4)
# print("Full results saved to grid_search_results.json")
