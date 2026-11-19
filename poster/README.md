# Poster

Project poster in the Duke theme, 24 x 24 in (square, 4 columns; the default) or 24 x 36 in portrait, built from saved run outputs (no numbers typed by
hand; key numbers are asserted against the phase reports). The layout, palette, fonts and chart
helper are adapted from the author's earlier Silent Second Opinion poster.

```bash
PYTHONPATH=scripts uv run python poster/build_poster.py                 # 24 x 24: poster.html and the PDF
PYTHONPATH=scripts uv run python poster/build_poster.py --size=24x36     # the portrait version
```

- `medlens-honesty-poster-24x24.pdf`: vector PDF for printing (square).
- `medlens-honesty-poster-24x36.pdf`: the earlier portrait version (same content).
- `fonts/`: EB Garamond and Open Sans (SIL Open Font License 1.1, via Google Fonts). Add the OFL
  text for each family if this repository is made public.
- `assets/duke-shield.png`: see `assets/README.md`. Its provenance and usage terms are unverified;
  confirm the use is acceptable before printing, or replace it with artwork from Duke's brand site.
