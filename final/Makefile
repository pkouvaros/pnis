TEX_NAME = main
LATEX = pdflatex
BIBTEX = bibtex
all:
	${LATEX} ${TEX_NAME}.tex
	${BIBTEX} ${TEX_NAME}
	${LATEX} ${TEX_NAME}.tex
	${LATEX} ${TEX_NAME}.tex
	rm ${TEX_NAME}.log ${TEX_NAME}.blg  ${TEX_NAME}.aux ${TEX_NAME}.bbl
