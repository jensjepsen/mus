"""
Extract the version number from a pyproject.toml file.
"""
import sys
import tomllib

def extract_version(pyproject_path):
    """Extract the version from a pyproject.toml file."""
    try:
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        # Try to find version in different possible locations
        if "project" in pyproject_data and "version" in pyproject_data["project"]:
            return pyproject_data["project"]["version"]
        elif "tool" in pyproject_data and "poetry" in pyproject_data["tool"] and "version" in pyproject_data["tool"]["poetry"]:
            return pyproject_data["tool"]["poetry"]["version"]
        else:
            print("Version not found in pyproject.toml")
            return None
    except FileNotFoundError:
        print(f"File not found: {pyproject_path}")
        return None
    except Exception as e:
        print(f"Error parsing pyproject.toml: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pyproject_path = sys.argv[1]
    else:
        pyproject_path = "pyproject.toml"  # Default to current directory
    
    version = extract_version(pyproject_path)
    if version:
        print(f"{version}", end="", flush=True)