# eegdash.viz.identity module

EEGDash data-visualization identity helpers.

The visual identity centres on the **Data Rail** — a thin EEGDash-blue
horizontal line near the top of every figure, with a short EEGDash-orange
pulse segment at its left edge. Tutorials call `use_eegdash_style()`
once at the top, then `style_figure()` per figure to get the rail,
title/subtitle/source band, and consistent spines/grids/ticks across
every axes.

<!-- !! processed by numpydoc !! -->

### eegdash.viz.identity.chance_line(ax, level: [float](https://docs.python.org/3/library/functions.html#float), , label: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'chance')

Add a horizontal dashed reference line at `level`.

<!-- !! processed by numpydoc !! -->

### eegdash.viz.identity.get_eegdash_palette(n: [int](https://docs.python.org/3/library/functions.html#int)) → [list](https://docs.python.org/3/library/stdtypes.html#list)[[str](https://docs.python.org/3/library/stdtypes.html#str)]

Return `n` colors from the EEGDash data-viz palette.

<!-- !! processed by numpydoc !! -->

### eegdash.viz.identity.style_figure(fig, , title: [str](https://docs.python.org/3/library/stdtypes.html#str), subtitle: [str](https://docs.python.org/3/library/stdtypes.html#str) = '', source: [str](https://docs.python.org/3/library/stdtypes.html#str) = '', data_rail: [bool](https://docs.python.org/3/library/functions.html#bool) = True, grid_axis: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'y') → [None](https://docs.python.org/3/library/constants.html#None)

Apply the EEGDash identity to every axes in `fig`.

* **Parameters:**
  * **fig** ([*matplotlib.figure.Figure*](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure)) – Target figure.
  * **title** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Short, 1-line figure title (top-left).
  * **subtitle** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Dataset/task/split context line.
  * **source** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Provenance footer (italic, bottom-left).
  * **data_rail** (bool, default `True`) – Attach the Data Rail.
  * **grid_axis** ( *{"y"* *,*  *"x"* *,*  *"both"* *,*  *"none"}*) – Grid orientation.

<!-- !! processed by numpydoc !! -->

### eegdash.viz.identity.use_eegdash_style() → [None](https://docs.python.org/3/library/constants.html#None)

Configure matplotlib (and seaborn if installed) for EEGDash plots.

<!-- !! processed by numpydoc !! -->
