TEX_DIR  := build/llm-core-architecture
TEX_FILE := $(TEX_DIR)/llm-core-architecture.tex
OUTPUT   := llm-core-architecture.pdf
ARCHIVE  := archive

.PHONY: all clean

all: $(OUTPUT)

$(OUTPUT): $(TEX_FILE)
	pdflatex -output-directory=$(TEX_DIR) $(TEX_FILE)
	pdflatex -output-directory=$(TEX_DIR) $(TEX_FILE)
	mv $(TEX_DIR)/llm-core-architecture.pdf $(OUTPUT)
	@mkdir -p $(ARCHIVE)
	ts=$$(date -r $(OUTPUT) +%Y-%m-%d_%H%M%S); \
	cp $(OUTPUT) $(ARCHIVE)/llm-core-architecture-$$ts.pdf; \
	echo "Saved archived document at $(ARCHIVE)/llm-core-architecture-$$ts.pdf"

clean:
	rm -f $(TEX_DIR)/*.aux $(TEX_DIR)/*.log $(TEX_DIR)/*.toc \
	      $(TEX_DIR)/*.out $(TEX_DIR)/*.nav $(TEX_DIR)/*.snm
