help: ## Show this help.
	@grep -E '^[a-zA-Z%_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install python requirements
	@make setup-requirements

setup-requirements: 
	@pip install wheel setuptools torch==2.2.1
	@pip install -r requirements.txt

test: ## Run all tests
	@python3 -m pytest tests/

remove-unused-imports: ## Remove unused imports
	@autoflake --in-place --remove-all-unused-imports -r graphphysics/ --exclude venv,node_modules

check-black: ## check black
	@black graphphysics/ --check

check-isort: ## check isort with black profile
	@isort graphphysics/ --profile black --check-only

lint: ## Remove unused imports, run linters Black and isort
	@make remove-unused-imports && isort graphphysics/ --profile black && black .