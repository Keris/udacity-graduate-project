THEMES_DIR = themes
STYLES_DIR = stylesheets
FONTS_DIR = fonts
STYLESHEET = github.css

ADOCS = *.adoc

.PHONY: all

all: html pdf

html:
	asciidoctor -r asciidoctor-diagram ${ADOCS}

.PHONY: pdf

pdf:
	asciidoctor-pdf -a pdf-stylesdir=${THEMES_DIR} -a pdf-style=custom -a pdf-fontsdir=${FONTS_DIR} ${ADOCS}

.PHONY: clean

clean:
	-rm -f *.pdf *.html
