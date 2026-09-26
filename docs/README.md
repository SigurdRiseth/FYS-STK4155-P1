# docs/

The project report: IEEE-format LaTeX source and its compiled PDF,
`main.pdf`, which is the final deliverable and the only large binary
artifact committed here.

```
docs/
├── main.tex          # document class, packages, title/author, section includes
├── main.pdf           # compiled report (committed — this is the deliverable)
├── references.bib     # bibliography (biblatex/biber)
├── sections/           # one file per report section
│   ├── 01-introduction.tex … 05-conclusion.tex
│   └── appendices/     # incl. 0a-ai-usage.tex, the LLM-usage disclosure
└── figures/            # generated PDFs/tables — gitignored, see figures/README.md
```

Build the PDF with:

```bash
make report     # regenerates figures (if stale) then runs latexmk
```

or, for just a LaTeX-only rebuild once figures already exist:

```bash
cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
make clean-report   # remove LaTeX build artifacts (not main.pdf itself)
```

Figures and tables under `figures/` are never edited by hand or committed
individually — they're regenerated from `scripts/`, from the numbers in
`data/results/`, so the report always reflects the current code. See
`figures/README.md`.
