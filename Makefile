PYTHON ?= python

.PHONY: help data quick run figures test all

help:
	@echo "make data     build the prompt set (AdvBench sample + paraphrases)"
	@echo "make quick    ~1 min smoke run into results/quick"
	@echo "make run      full experiment into results/"
	@echo "make figures  regenerate figures/ from results/"
	@echo "make test     fast unit tests (add SLOW=1 to include the gpt2 checks)"

data:
	$(PYTHON) -m src.data

quick:
	$(PYTHON) -m src.run_experiment --quick
	$(PYTHON) -m src.figures --results results/quick --figures figures/quick

run:
	$(PYTHON) -m src.run_experiment

figures:
	$(PYTHON) -m src.figures

test:
ifdef SLOW
	$(PYTHON) -m pytest -q -m slow
else
	$(PYTHON) -m pytest -q
endif

all: data run figures
