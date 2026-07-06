.PHONY: help install train run test lint clean

help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies"
	@echo "  make train     - Train the ML model"
	@echo "  make run       - Run the Flask API"
	@echo "  make test      - Run unit tests"
	@echo "  make lint      - Check code quality"
	@echo "  make clean     - Remove generated files"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	python train.py

run:
	python app.py

test:
	pytest -v

lint:
	pylint app.py train.py

clean:
	python -c "from pathlib import Path; import shutil; Path('models/model.pkl').unlink(missing_ok=True); shutil.rmtree('.pytest_cache', ignore_errors=True); shutil.rmtree('__pycache__', ignore_errors=True); shutil.rmtree('tests/__pycache__', ignore_errors=True)"