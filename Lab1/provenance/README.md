# Evidence provenance

The experiment was performed on the supplied Raspberry Pi 4B. Its completed project was exported to OMEN as `lab1-complete-20260910.tar.gz` with SHA-256:

```text
1b78458bad8a645818dd3c6b9f829ca8d2b229126d6e58c76d31eff70991b856
```

`pi_export_sha256.json` is the original 38-file SHA-256 manifest generated on the Pi before that export. Every measurement, console log, camera image, classification result, and Python source file covered by it is preserved byte-for-byte.

After export, the report was reformatted on OMEN using XeLaTeX to match the requested plain course-document style. Only `results/Lab1_Report.pdf` and the accompanying `results/report.md` document note were revised. Their old and current checksums are recorded in `document_revisions.json`. `results/evidence_sha256.json` records the final hashes, including the LaTeX source. This typesetting change does not alter any latency observation or statistic.

The optional `results/model-graph-analysis.json` contains static ONNX graph analysis carried out on OMEN, not inference timing. Its Conv-only MAC estimates are distinct from the measured Raspberry Pi latencies. The report explicitly distinguishes them.

The teacher's PDF instructions, account credentials, Wi-Fi settings, and OMEN troubleshooting files are not part of this repository. Upstream attribution and model/sample source manifests are retained.
