# Adpicker CPEX  ODAP
AdPicker project for CPEx client using _ODAP light_ for managing features and feature store in more comfortable way.

## Structure

```
.
├── .github
│   └── workflows ... deployment and PR workflows
├── _orchestration  ...  ODAP orchestration notebooks, part of the framework
├── features  ...  notebooks for feature calculation
│   ├── stage1
│   │   ├── interests
│   │   └── web_behaviour
│   ├── stage2
│   │   ├── edu
│   │   ├── income
│   │   ├── lookalikes
│   │   └── sociodemo
│   └── stage3
│       └── abcde_score.py
├── init  
│   ├── odap.py  ...  init notebook for ODAP
│   └── requirements_pylint.txt  ...   and pylint requirements
├── src
│   ├── bronze
│   │   ├── piano  ...  Piano raw -> bronze
│   │   └── test_notebook.py  ...  TODO remove
│   ├── config  ...  actual paths and pipeline configs!
│   │   └── config.yaml  ...  do not read directly, use src.utils.read_config
│   ├── monitoring  ...  regularly run monitoring notebooks
│   ├── schemas  ... schemas required in various notebooks
│   ├── silver   ...   silver layer
│   │   ├── piano
|   │   |   └─ piano_user_segments.py  ...  download piano segments
│   │   └── sdm  ...   Smart Data Model
│   ├── solutions  ...  ML and other solutions
│   │   ├── education  ...  education scores
│   │   └── income  ...  income scores
│   ├── tools  ...  various INTERACTIVE notebooks
│   │   └── interests  ... interest creation and management
│   └── utils  ...  library of useful functions
│       ├── helper_functions_defined_by_user  ...  TODO refactor
│       ├── processing_pipelines  ...  ???
│       ├── config_parser.py  ... utils for finding and parsing config
│       └── read_config.py  ...  import config from here!
├── use_cases  ...  part of ODAP framework
├── config.yaml  ...  ODAP framework config, do not touch!
├── .pylintrc  ...  PyLint config, its annoying but do not touch
└─ README.md  ... this readme yayyy!
```

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

You can also face issue with running `pylint` locally with error that looks similar to `[Errno 2] No such file or directory: '__init__.py' (parse-error)` - it's caused by running `pylint` on whole package/project where recursive run can be tricky (for `pylint` at least). So to solve it, there's an argument `--recursive=y` that you add to runnable command that in it's final form can look like `python -m pylint --recursive=y .` and it should work like a charm.

# Config
Why? 
Using [OmegaConf](https://omegaconf.readthedocs.io/en/latest/index.html) makes it easier to access values from nested yaml it also provides string interpolation enabling you to use
values defined within the `config.yaml` itself and environment variables.

How?
The `src/utils/read_config.py` file returns the config file as an object 
stored in the variable `config`.

Usage:
Import the config object from the read config file
`from src.utils.read_config import config`

To access the value stored at a certain key OmegaConf provides object 
style access of dictionary elements.
```yaml
paths:
  some_key : "some_path"
```

To access the value stored at the key "some_key" you can use the following example
`path = config.paths.some_key` 

Add new value to the config object:
Go to `src/config/config.yaml` and edit the yaml file, the changes will be reflected in
the config object when you import it anew (this might require detaching and reattaching running notebook).