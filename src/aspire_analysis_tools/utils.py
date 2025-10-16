
def read_make_config(filename: str) -> dict:
    """Read a simple key=value config file, ignoring comments and blank lines."""
    config = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config