# Overleaf Setup Instructions

## Files Included

1. **Research_Proposal_Assignment1.tex** - Main LaTeX document
2. **references.bib** - Bibliography file in BibTeX format

## How to Upload to Overleaf

### Method 1: Upload Individual Files
1. Go to [Overleaf.com](https://www.overleaf.com) and create a new project
2. Click "New Project" → "Blank Project"
3. Name your project (e.g., "Research Proposal Assignment 1")
4. Upload `Research_Proposal_Assignment1.tex` as the main file
5. Upload `references.bib` to the same project folder
6. In Overleaf, set `Research_Proposal_Assignment1.tex` as the main document (if not automatically detected)

### Method 2: Upload as ZIP
1. Create a ZIP file containing both `.tex` and `.bib` files
2. In Overleaf, click "New Project" → "Upload Project"
3. Upload the ZIP file
4. Overleaf will extract and compile automatically

## Compilation

1. Click "Recompile" button (or press Ctrl/Cmd + Enter)
2. If you get compilation errors, check:
   - Both `.tex` and `.bib` files are in the same folder
   - The main `.tex` file is set correctly
   - All packages are available (they should be - using standard packages)

## Customization

### To Change Bibliography Style:
In the `.tex` file, find:
```latex
\bibliographystyle{agsm}
```

Alternative Harvard styles you can use:
- `agsm` (Australian Government Style Manual - current)
- `dcu` (Dublin City University)
- `kluwer`
- `chicago` (Chicago Manual of Style)

Just change the style name in the document.

### To Add Author Information:
Uncomment and fill in:
```latex
\author{Your Name\\
Your Institution\\
\href{mailto:your.email@example.com}{your.email@example.com}}
```

### To Add Date:
Uncomment:
```latex
\date{\today}
```

## Notes

- The document uses `natbib` package for Harvard-style citations
- Citations in text use `\citep{}` for parenthetical citations
- Table of contents is automatically generated
- All sections are properly numbered
- Hyperlinks are colored blue for links and citations

## Troubleshooting

**Error: "Bibliography not found"**
- Make sure `references.bib` is in the same folder as the main `.tex` file
- Compile twice: first to generate references, second to include them

**Error: "Package natbib Error"**
- The `natbib` package should be available in Overleaf
- If issues persist, try changing `\bibliographystyle{agsm}` to `\bibliographystyle{dcu}`

**Table formatting issues:**
- The longtable should automatically break across pages
- If it doesn't fit, you can reduce font size or adjust column widths

## Tips

1. Compile frequently to catch errors early
2. Use Overleaf's word count feature (found in the menu)
3. Check the PDF preview regularly
4. Save your work often (Overleaf auto-saves but it's good practice)

