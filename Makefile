# Paper artifacts, rebuilt from results/ only.
#   make figs      the five data figures in paper/figs/
#   make numbers   paper/numbers.tex and paper/numbers_table.md
PY := .venv/bin/python
FIGS := wave heads removal winner leftover

.PHONY: figs numbers $(FIGS)

figs: $(FIGS)

$(FIGS):
	$(PY) paper/figs/$@.py

numbers:
	$(PY) scripts/make_numbers.py
