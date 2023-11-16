# adpicker-cpex-odap
AdPicker project for CPEx client using _ODAP light_ for managing features and feature store in more comfortable way.

## Github actions
There are few automatic [Github Actions](https://github.com/DataSentics/adpicker-cpex-odap/actions), maily for following procedures:
- Pylint Formatting check - which use package `pylint` from Pypi [here](https://pypi.org/project/pylint/) that works as static (also possible to run
it local - more information can be found [here](https://pylint.readthedocs.io/en/latest/user_guide/installation/index.html)) Python code analyzer
- Black Formatter - which use package `black` from Pypi [here](https://pypi.org/project/black/) that automatically format all Python
files according to [PEP8](https://realpython.com/python-pep8/) standard (also possible to run it locally - more information can be found [here](https://black.readthedocs.io/en/stable/getting_started.html))
- DBX Deployment - Databricks deployment using `cli` repository [on Github](https://github.com/databricks/cli) - that allows you to connect
into your Databricks workspace using command line commands and interact with various APIs. If you would like to run it locally/directly from your code, it's necessary or suggested to use repository with [Databricks SDK for Python](https://github.com/databricks/databricks-sdk-py).

## Suggestions
However it can be nice to use only online tools for checking formatting and formatting itself, it's suggested to check your code locally with Pylint locally for any discrepancies that can cause 
non-functionality of your code and can be easily missed. This can be done by using command `pylint <path to folder or file>` in terminal.

Same idea goes with `black` formatting, if you are not sure which files will be reformatted you can check them with command `black --check --verbose <path to folder or file>` (using masks, relative or absolute path is possible) in terminal.

For both packages / functions the default path (when you not specify it differently) is `.` which tooks your current folder as root one and recursively go through all folders and files.

### Tip
There is quite a high possibility that showed commands for local run  wouldn't work, in that case try to use them with "prefix" `python -m` so they will look like `python -m pylint <path to folder or file>` and `python -m black --check --verbose <path to folder or file>`
