.PHONY: install clean test format lint prepare-data train evaluate

install:
	pip install -e .
	pip install -r requirements.txt
	pre-commit install

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist

test:
	pytest tests/ -v --cov=. --cov-report=html

format:
	black .
	isort .

lint:
	flake8 .
	mypy .

prepare-data:
	python scripts/prepare_data.py --config configs/dataset.yaml

train:
	python scripts/train.py --config configs/train.yaml

evaluate:
	python scripts/evaluate.py --config configs/eval.yaml

verify:
	python scripts/verify_run.py