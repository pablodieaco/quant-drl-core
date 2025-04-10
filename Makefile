# Makefile

PACKAGE_NAME = quant-drl-core
IMPORT_NAME = quant_drl

install-dev:
	@echo "Instalando el paquete en modo editable..."
	pip install -e .[dev]

# Delete dist, build y archivos temporales
clean:
	@echo "Limpiando archivos temporales..."
	rm -rf build/ dist/ *.egg-info

# Build package (wheel y sdist)
build: clean
	@echo "Construyendo paquete..."
	python -m build

# Publish in PyPI (production)
publish: build
	@echo "Subiendo a PyPI..."
	twine upload dist/*

# Publish in TestPyPI (testing)
publish-test: build
	@echo "Subiendo a TestPyPI..."
	twine upload --repository testpypi dist/*

# Install build y twine
setup-tools:
	pip install build twine

# Test install from TestPyPI
test-install:
	pip install --index-url https://test.pypi.org/simple $(PACKAGE_NAME)
	python -c "import $(IMPORT_NAME); print($(IMPORT_NAME).__version__)"


.PHONY: install-dev clean build publish publish-test setup-tools test-install
