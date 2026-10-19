# Final report

`report.html` is the editable source used for the submitted PDF. `final_report.tex` is an equivalent self-contained two-column LaTeX version suitable for Overleaf or a local TeX installation. Both contain the token `{{COMMIT_HASH}}`; replace it with the intended repository commit before compiling.

From the assignment directory, build the PDF with the current Git commit:

```powershell
powershell -ExecutionPolicy Bypass -File report/build_report.ps1
```

To render a specific recorded commit instead:

```powershell
powershell -ExecutionPolicy Bypass -File report/build_report.ps1 -CommitHash 0123456789abcdef
```

The output is `report/final_report.pdf`. The build requires Google Chrome or Microsoft Edge and uses only local files. The PDF is ignored by Git because it embeds the final repository commit hash and is submitted separately.

To use the LaTeX version, upload the complete repository to Overleaf, set `report/final_report.tex` as the main document, replace `{{COMMIT_HASH}}` with the final hash, and compile with pdfLaTeX. The figure paths assume the existing repository directory structure.
