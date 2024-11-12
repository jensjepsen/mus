PYTHON = $(shell `pwd`/.venv/bin/python)
venv/touchfile: requirements.txt
	python3 -m venv venv
	$(PYTHON) -m pip install -r requirements.txt
	touch venv/touchfile