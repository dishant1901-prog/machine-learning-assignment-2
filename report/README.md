# Final report

`report.html` is the editable two-column report source. It contains the token `{{COMMIT_HASH}}`, which is replaced only in the rendered copy so the source remains reusable.

From the assignment directory, build the PDF with the current Git commit:

```powershell
powershell -ExecutionPolicy Bypass -File report/build_report.ps1
```

To render a specific recorded commit instead:

```powershell
powershell -ExecutionPolicy Bypass -File report/build_report.ps1 -CommitHash 0123456789abcdef
```

The output is `report/final_report.pdf`. The build requires Google Chrome or Microsoft Edge and uses only local files. The PDF is ignored by Git because it embeds the final repository commit hash and is submitted separately.
