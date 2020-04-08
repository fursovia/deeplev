
.PHONY: check-codestyle codestyle

codestyle:
	pre-commit run --all-files

check-codestyle:
	python3 scripts/verify.py --checks flake8
	python3 scripts/verify.py --checks mypy
	python3 scripts/verify.py --checks black
