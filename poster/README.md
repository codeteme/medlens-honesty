# Poster

24 x 36 in portrait poster in the Duke theme, built from saved run outputs (no numbers typed by
hand; key numbers are asserted against the phase reports). The layout, palette, fonts and chart
helper are adapted from the author's earlier Silent Second Opinion poster.

```bash
PYTHONPATH=scripts uv run python poster/build_poster.py     # writes poster.html and the PDF
```

- `medlens-honesty-poster-24x36.pdf`: vector PDF for printing.
- `fonts/`: EB Garamond and Open Sans (SIL Open Font License 1.1, via Google Fonts). Add the OFL
  text for each family if this repository is made public.
- `assets/duke-shield.png`: see `assets/README.md`. Its provenance and usage terms are unverified;
  confirm the use is acceptable before printing, or replace it with artwork from Duke's brand site.
