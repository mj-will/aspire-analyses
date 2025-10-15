import json

def extract_prior_bounds(prior_string: str) -> dict:
    """Extract the prior bounds from a pesummary prior string."""
    # Parse JSON string into a dictionary
    data = json.loads(prior_string)
    
    prior_bounds = {}
    for param, details in data.items():
        # Skip metadata keys
        if not isinstance(details, dict) or "kwargs" not in details:
            continue
        
        kwargs = details["kwargs"]
        minimum = kwargs.get("minimum")
        maximum = kwargs.get("maximum")
        
        if minimum is not None and maximum is not None:
            prior_bounds[param] = (minimum, maximum)
    
    return prior_bounds