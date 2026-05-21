# ═══════════════════════════════════════════════════════════
#   ML Dashboard  environment
#  Usage:
#    make        → create venv + install deps + run dashboard
#    make install→ create venv + install deps only
#    make run    → run dashboard (venv must exist)
#    make clean  → remove venv completely
# ═══════════════════════════════════════════════════════════

ENV_NAME  = dashboard_env
PYTHON    = python3
VENV      = $(ENV_NAME)/bin/activate
PIP       = $(ENV_NAME)/bin/pip
RUN       = $(ENV_NAME)/bin/python

#  install + run 
all: install run

# create venv and install requirements
install: $(VENV)

$(VENV): requirements.txt
	@echo ""
	@echo "┌─────────────────────────────────────────────┐"
	@echo "│  Creating virtual environment: $(ENV_NAME)  │"
	@echo "└─────────────────────────────────────────────┘"
	$(PYTHON) -m venv $(ENV_NAME)
	$(PIP) install --upgrade pip --quiet
	$(PIP) install --no-cache-dir -r requirements.txt
	@echo ""
	@echo "  Environment ready!"

#  run the dashboard 
run:
	@echo ""
	@echo "┌─────────────────────────────────────────────┐"
	@echo "│  Starting Dashboard → http://127.0.0.1:8050 │"
	@echo "└─────────────────────────────────────────────┘"
	@echo ""
	$(RUN) ml_dashboard.py

#remove the virtual environment
clean:
	@echo ""
	@echo "   Removing $(ENV_NAME)..."
	rm -rf $(ENV_NAME)
	@echo "  Clean done. Your system packages are untouched."
	@echo ""

.PHONY: all install run clean
