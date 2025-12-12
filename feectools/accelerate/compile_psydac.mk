# Based on https://gitlab.mpcdf.mpg.de/struphy/struphy/-/blob/devel/src/struphy/compile_struphy.mk
#--------------------------------------
# CONFIGURATION
#--------------------------------------

PYTHON  := python3
SO_EXT  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
LIBDIR  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
psydac_path := $(shell $(PYTHON) -c "import feectools as _; print(_.__path__[0])")

# Arguments to this script are: 
PSYDAC_SOURCES := $(sources)
FLAGS := --libdir $(LIBDIR) $(flags) 
FLAGS_openmp := $(flags_openmp)

#--------------------------------------
# SOURCE FILES 
#--------------------------------------

SOURCES := $(PSYDAC_SOURCES)

OUTPUTS := $(SOURCES:.py=$(SO_EXT))

#--------------------------------------
# PYCCELIZE
#--------------------------------------

.PHONY: all
all: $(OUTPUTS)

# %$(SO_EXT) : %.py $$(shell $$(PYTHON)) $$(psydac_path)/dependencies.py $$@)
.SECONDEXPANSION:
%$(SO_EXT): %.py

	@echo "Building $@"
	@echo "from dependencies:"
	@for dep in $^ ; do \
		echo $$dep ; \
    done
	pyccel compile -v $(FLAGS)$(FLAGS_openmp) $<
	@echo ""

#--------------------------------------
# CLEAN UP
#--------------------------------------

.PHONY: clean
clean:
	rm -rf $(OUTPUTS)
	find $(psydac_path)/ -type d -name '__pyccel__' -prune -exec rm -rf {} \;
	find $(psydac_path)/ -type d -name '__pycache__' -prune -exec rm -rf {} \;
	find $(psydac_path)/ -type f -name '*.lock' -delete