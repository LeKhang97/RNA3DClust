VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip3

run: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	pyclean .

.PHONY: all run clean
