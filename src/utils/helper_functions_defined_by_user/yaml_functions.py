import os
import yaml
import re


def find_and_replace_percent_substrings(source_string):
    pattern = r"%(.*?)%"
    matches = re.findall(pattern, source_string)

    for find in matches:
        try:
            source_string = re.sub(f"%{find}%", os.getenv(find), source_string)
        except TypeError as e:
            raise BaseException(
                    "Could not replace special substring; check if substring is defined in cluster environment variables."
            ) from e

    return source_string


# function to get a value from a .yaml file by passing name of keys
def get_value_from_yaml(*keys):
    """
    This function gets the value (key : value format) from a .yaml file

    Parameters:
    *keys - a variable number of parameters representings the keys you have to enter to access a value

    Return value:
    a string representing the wanted value from .yaml file
    """

    # Get the path to the directory where functions.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the config.yaml file
    yaml_path = os.path.join(current_dir, "../../config/config.yaml")

    with open(yaml_path, "r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)

        for key in keys:
            if key in data:
                data = data[key]
            else:
                raise KeyError(f"Key '{key}' not found in the YAML file.")

        if isinstance(data, str):
            final_string = find_and_replace_percent_substrings(data)
            return final_string

        return data


def environment_tag(old_tag, new_tag):
    # Read the environment variable from the cluster
    # env_variable_value = os.environ.get("APP_ENV")

    # Path to the .yaml file
    # absolute_yaml_path = os.getcwd() + "/src/adpickercpex/_config/config.yaml"
    yaml_path = "/Workspace/Users/nicolellaurian.anistoroaei@datasentics.com/cnfg.yaml"

    # Read the contents of the .yaml file
    with open(yaml_path, "r", encoding="utf-8") as yaml_file:
        yaml_content = yaml_file.read()

    # Replace the placeholder with the environment variable value
    yaml_content_with_replacement = yaml_content.replace(f"{old_tag}", f"{new_tag}")

    # Save the modified content back to the same .yaml file
    with open(yaml_path, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(yaml_content_with_replacement)
