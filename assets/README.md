# Visual assets

The banner adapts the layout supplied by the project owner. Its scatter is rebuilt from `reports/held_out_predictions.csv`, not decorative points. Marker area encodes overlapping record count. The five zero-grade cases occupy four unique positions: (0, 0), (0, 7), (0, 8), and (0, 9), with two records at the last position.

Run `python scripts/build_assets.py` after regenerating reports to rebuild `banner.svg` and `social-preview.png`. The social preview is 1280 × 640 pixels. The app screenshot is captured from the running app; it is not a mockup. Dataset attribution and CC BY 4.0 terms are in `data/README.md`.
