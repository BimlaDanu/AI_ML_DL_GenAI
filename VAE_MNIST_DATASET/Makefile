# ═══════════════════════════════════════════════════════════
#  Makefile —  Torch Environment
#  TORCH ENV:
#    make torch        → create torch_env + install all deps
#    make torch-kernel → register torch_env as Jupyter kernel
#    make torch-test   → verify torch, torchvision, cv2 imports
#    make torch-clean  → remove torch_env
#    make clean-all    → remove both environments
# ═══════════════════════════════════════════════════════════

PYTHON        = python3
TORCH_ENV     = torch_env
TORCH_PIP     = $(TORCH_ENV)/bin/pip
TORCH_RUN     = $(TORCH_ENV)/bin/python
TORCH_VENV    = $(TORCH_ENV)/bin/activate

# ───────────────────────────────────────────────────────────
#  TORCH ENV
# ───────────────────────────────────────────────────────────

torch: $(TORCH_VENV)

$(TORCH_VENV): requirements_torch.txt
	@echo ""
	@echo "┌──────────────────────────────────────────────────┐"
	@echo "│  Creating: $(TORCH_ENV)                          │"
	@echo "│  Installing: torch, torchvision, opencv,         │"
	@echo "│              numpy 1.26.4, pandas, sklearn       │"
	@echo "└──────────────────────────────────────────────────┘"
	$(PYTHON) -m venv $(TORCH_ENV)
	$(TORCH_PIP) install --upgrade pip --quiet
	$(TORCH_PIP) install --no-cache-dir -r requirements_torch.txt
	@echo ""
	@echo "  torch_env ready!"
	@echo "    Activate : source $(TORCH_ENV)/bin/activate"
	@echo "    Test     : make torch-test"
	@echo "    Jupyter  : make torch-kernel"

torch-kernel:
	@echo "  Registering torch_env as Jupyter kernel..."
	$(TORCH_RUN) -m ipykernel install --user \
		--name=torch_env \
		--display-name "Python (torch_env)"
	@echo "  Kernel registered!"
	@echo "    In Jupyter: Kernel → Change Kernel → Python (torch_env)"

torch-test:
	@echo ""
	@echo "  Testing torch_env imports..."
	$(TORCH_RUN) -c "\
import numpy; print('numpy      :', numpy.__version__); \
import pandas; print('pandas     :', pandas.__version__); \
import sklearn; print('sklearn    :', sklearn.__version__); \
import torch; print('torch      :', torch.__version__); \
import torchvision; print('torchvision:', torchvision.__version__); \
import cv2; print('opencv     :', cv2.__version__); \
print(''); \
print('CUDA available:', torch.cuda.is_available()); \
print('MPS  available:', torch.backends.mps.is_available()); \
"
	@echo ""
	@echo "  All imports OK!"

torch-clean:
	@echo "   Removing $(TORCH_ENV)..."
	rm -rf $(TORCH_ENV)
	@echo "  torch_env removed. System packages untouched."

# ───────────────────────────────────────────────────────────
#  CLEAN ALL
# ───────────────────────────────────────────────────────────

clean-all: clean torch-clean
	@echo "  All environments removed."

.PHONY: all install run clean torch torch-kernel torch-test torch-clean clean-all
