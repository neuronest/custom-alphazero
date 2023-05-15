#!/usr/bin/env bash

set -e

# Color
GREEN='\033[0;32m'
NC='\033[0m' # No Color
RED='\033[0;31m'

MIN_REQUIRED_PYTHON_VERSION=3.8.0
MAX_EXCLUDED_REQUIRED_PYTHON_VERSION=3.9.0
PATH_TO_PYTHON=$(which python3)

function green {
  printf "${GREEN}$@${NC}\n"
}

function red {
  printf "${RED}$@${NC}\n"
}

function help {
     echo -e \\n"Help documentation for ${GREEN}$0${NC}"\\n
     echo "This script allows to install project."
     echo "Command line switches are optional. The following switches are recognized."
     echo -e "By default, the script will take python from ${GREEN}$(which python3)${NC}"
     echo "Expected python version has to be between $MIN_REQUIRED_PYTHON_VERSION - $MAX_EXCLUDED_REQUIRED_PYTHON_VERSION"
     echo -e "${GREEN}-p${NC}  --Sets the path to python3"
     echo -e "${GREEN}-h${NC}  --Displays this help message. No further functions are performed."\\n
     exit 1
 }

function version {
  echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }';
  }

function check_python_version {
  CURRENT_PYTHON_VERSION=$("$@" -c "import sys; print('.'.join([str(s) for s in sys.version_info[:3]]))")
  if ! poetry --version > /dev/null; then
    echo "Poetry not found!"
    exit 1
  fi
  poetry self update
  if ! (($(version $CURRENT_PYTHON_VERSION) >= $(version $MIN_REQUIRED_PYTHON_VERSION)
  && $(version $CURRENT_PYTHON_VERSION ) < $(version $MAX_EXCLUDED_REQUIRED_PYTHON_VERSION))); then
    echo "Expected python version has to be between $MIN_REQUIRED_PYTHON_VERSION - $MAX_EXCLUDED_REQUIRED_PYTHON_VERSION"
    echo "But given: $CURRENT_PYTHON_VERSION"
    help
    exit 1
  fi
  if ! poetry env use $PATH_TO_PYTHON > /dev/null; then
    echo $(red Can not find python $CURRENT_PYTHON_VERSION);
    help
    exit 1
  fi
}

function delete_poetry_env {
  if ! [ -f "pyproject.toml" ]; then
    echo "pyproject.toml does not exist. Aborting..."
    exit 1
  fi
  poetry env remove $(basename $(poetry env info -p))
}


while getopts p:h option
   do
     case "${option}" in
       p) PATH_TO_PYTHON=$OPTARG;;
       h) help;;
       \?) echo -e "Argument number ${RED}$OPTIND${NC} is wrong";
         help
         exit 1;;
     esac
  done
 shift $(($OPTIND - 1))

check_python_version $PATH_TO_PYTHON
delete_poetry_env

poetry env use $PATH_TO_PYTHON
poetry run pip install --upgrade pip
poetry install --with dev

ENV=$(poetry env info -p)
ENV_BIN=$ENV/bin

$ENV_BIN/pre-commit install
$ENV_BIN/pre-commit install --hook-type commit-msg

echo "virtual environment: $(green $ENV)"
echo "activate environment: $(red poetry shell)"
