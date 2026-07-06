install:
	pip install --upgrade pip
	pip install -r requirements.txt

lint:
	pylint app.py train.py

test:
	pytest

run:
	python app.py