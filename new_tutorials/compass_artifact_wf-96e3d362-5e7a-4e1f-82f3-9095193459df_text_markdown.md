# Building an Evidence-Based Tutorial for `eegdash`: Validation Rubric, Best-Practice Guide, and Recommended Structure

This report synthesises (A) cognitive-science and computing-education research on tutorial design, (B) scientific-software documentation best practices (Diátaxis, sphinx-gallery/scikit-learn, MNE-Python, Braindecode), (C) neuroscience/EEG education research, (D) engagement/motivation literature, (E) a concrete validation rubric you can apply to an AI agent's draft, and (F) a recommended tutorial structure tailored to `eegdash`.

---

## TL;DR

- **`eegdash`** is the Python client for **EEG-DaSh**, an NSF-funded data-sharing archive for M/EEG/fNIRS/EMG/iEEG (a UCSD-SCCN × Ben-Gurion University collaboration). It is BIDS-first, returns Braindecode-/PyTorch-compatible datasets, and is sibling-in-spirit to MNE-Python and Braindecode rather than the "Plotly-Dash EEG visualiser." The natural model for tutorials is therefore the **scikit-learn / MNE-Python / Braindecode sphinx-gallery format**, where each example is a single executable `plot_*.py` file rendered into HTML + downloadable `.ipynb` + `.py`.
- The strongest evidence base for tutorial design converges on a few principles: **manage cognitive load** (Sweller; Paas; van Merriënboer), use **worked examples that fade into Parsons/completion problems** (Renkl; Sentance & Waite's PRIMM), **read code before writing code**, **keep one notebook = one narrative** (Rule et al. 2019), use **authentic data with a real research question**, prefer **participatory live-coding style narration** in prose (Nederbragt et al. 2020), and respect **Diátaxis** by keeping a *tutorial* learning-oriented and not mixing it with reference/how-to content.
- The rubric in §E below converts these findings into a checklist (structural, pedagogical, technical, engagement, domain) that you can paste into a review prompt or use as a PR-review checklist for the AI agent's output.

---

## Key Findings

1. **`eegdash` is a data-access library, not a Plotly Dash app.** Its `EEGDashDataset` returns `braindecode` datasets (i.e., PyTorch datasets), data are BIDS-compatible, and the project already publishes a sphinx-gallery at `eegdash.org/generated/auto_examples/`. Tutorials should therefore look and behave like MNE-Python / Braindecode examples, with explicit interoperability hooks to those libraries.
2. **Cognitive Load Theory is the central design lens.** Novices learn faster from worked examples than from problem-solving; split-attention, redundancy, and modality effects all argue for code + adjacent prose + figures in the same visual unit (notebook cells are ideal for this). The expertise-reversal effect warns that what helps a novice can bore an expert — so tutorials must be tagged by level and progressively fade scaffolding.
3. **The PRIMM cycle (Predict → Run → Investigate → Modify → Make)** and **faded Parsons / completion problems** are the best-validated structures for moving learners from reading to writing code in a tutorial setting.
4. **Notebook-specific evidence (Rule et al. 2019)** gives 10 concrete rules: tell a story, document the process, use cell divisions to highlight steps, modularise reused code, record dependencies, build pipelines, use version control, share data + environment, design for re-running top-to-bottom, and enable reader exploration.
5. **Diátaxis (Procida)** is now the de-facto framework for technical docs; an `eegdash` tutorial must be *learning-oriented*, must produce a successful, end-to-end concrete result, and must not drift into reference or how-to content.
6. **Live coding / participatory narration** improves engagement and process fidelity; in static notebooks this translates to explicit "what we are about to do / what just happened / why this matters" prose around each cell, plus deliberate exposure of small errors and how to recover.
7. **Authentic data + a real research question** (e.g., eyes-open vs eyes-closed classification, or HBN child resting-state spectra) drive intrinsic motivation per Self-Determination Theory and improve transfer to learners' own work.
8. **Neuroscience-specific guidance** (Cisotto & Chicco 2024 "Ten quick tips for clinical EEG"; Millman et al. 2018 on teaching computational reproducibility for neuroimaging; Neuromatch Academy) emphasises: explicit montage/reference handling, filtering choices and pitfalls, ICA/artefact awareness, BIDS metadata, FAIR data citation, and a project-based capstone.
9. **Reproducibility is non-negotiable.** Wilson et al. (2014, 2017), Sandve et al. (2013), Wilkinson et al. (2016, FAIR), and Rule et al. (2019) all converge: pin versions, seed RNGs, restart-and-run-all, deposit data with DOIs, and make the notebook the single source of truth.

---

## Details

### A. Educational and cognitive-science literature

**Cognitive Load Theory and its sub-effects**
- Sweller, J. (1988). Cognitive load during problem solving: Effects on learning. *Cognitive Science* 12(2): 257–285. https://doi.org/10.1207/s15516709cog1202_4
- Sweller, J., van Merriënboer, J. J. G., & Paas, F. (1998 / 2019). Cognitive architecture and instructional design. *Educational Psychology Review* 10(3): 251–296 (and the 20-years-later update, *Educational Psychology Review* 31: 261–292, 2019). https://doi.org/10.1007/s10648-019-09465-5
- Chandler, P., & Sweller, J. (1991). Cognitive load theory and the format of instruction. *Cognition and Instruction* 8(4): 293–332. (split-attention)
- Sweller, J., & Cooper, G. A. (1985). The use of worked examples as a substitute for problem solving in learning algebra. *Cognition and Instruction* 2(1): 59–89. (worked-example effect)
- Kalyuga, S., Ayres, P., Chandler, P., & Sweller, J. (2003). The expertise reversal effect. *Educational Psychologist* 38(1): 23–31. https://doi.org/10.1207/S15326985EP3801_4
- Paas, F., & van Merriënboer, J. J. G. (1994). Variability of worked examples and transfer of geometrical problem-solving skills: a cognitive-load approach. *Journal of Educational Psychology* 86(1): 122–133.

**Implication for `eegdash`:** Keep prose adjacent to the code that it explains (avoid split-attention). Use complete worked examples for novices; create separate "advanced" galleries that omit redundant explanation for experienced users (managing the expertise-reversal effect).

**Programming education / computing-education research**
- Brown, N. C. C., & Wilson, G. (2018). Ten quick tips for teaching programming. *PLOS Computational Biology* 14(4): e1006023. https://doi.org/10.1371/journal.pcbi.1006023
- Wilson, G. (2019). Ten quick tips for delivering programming lessons. *PLOS Computational Biology* 15(10): e1007433. https://doi.org/10.1371/journal.pcbi.1007433
- Nederbragt, A., Harris, R. M., Hill, A. P., & Wilson, G. (2020). Ten quick tips for teaching with participatory live coding. *PLOS Computational Biology* 16(9): e1008090. https://doi.org/10.1371/journal.pcbi.1008090
- Sentance, S., & Waite, J. (2017). PRIMM: Exploring pedagogical approaches for teaching text-based programming in school. *Proc. WiPSCE '17* (ACM). https://doi.org/10.1145/3137065.3137072
- Sentance, S., Waite, J., & Kallia, M. (2019). Teaching computer programming with PRIMM: a sociocultural perspective. *Computer Science Education* 29(2-3): 136–176. https://doi.org/10.1080/08993408.2019.1608781
- Weinman, N., Fox, A., & Hearst, M. A. (2021). Improving instruction of programming patterns with Faded Parsons Problems. *Proc. CHI 2021*. https://doi.org/10.1145/3411764.3445228
- Hou, X., Ericson, B. J., & Wang, X. (2024). Understanding the effects of using Parsons problems to scaffold code writing for students with varying CS self-efficacy levels. *Proc. Koli Calling 2023* / arXiv:2311.18115.
- Rubin, M. J. (2013). The effectiveness of live-coding to teach introductory programming. *Proc. SIGCSE 2013*. https://doi.org/10.1145/2445196.2445388
- Renkl, A. (2014). Toward an instructionally oriented theory of example-based learning. *Cognitive Science* 38(1): 1–37.

**Software/Data Carpentry pedagogy**
- Wilson, G. (2006). Software Carpentry: Getting scientists to write better code by making them more productive. *Computing in Science & Engineering* 8(6): 66–69.
- Wilson, G. (2016). Software Carpentry: Lessons learned. *F1000Research* 3: 62. https://doi.org/10.12688/f1000research.3-62.v2
- Devenyi, G. A., et al. (2018). Ten Simple Rules for Collaborative Lesson Development. *PLOS Computational Biology* 14(3): e1005963.
- Wilson, G. (2019). Ten Quick Tips for Creating an Effective Lesson. *PLOS Computational Biology* 15(4): e1006915.

**Notebooks as educational artefacts**
- Rule, A., Birmingham, A., Zuniga, C., Altintas, I., Huang, S.-C., Knight, R., Moshiri, N., Nguyen, M. H., Rosenthal, S. B., Pérez, F., & Rose, P. W. (2019). Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks. *PLOS Computational Biology* 15(7): e1007007. https://doi.org/10.1371/journal.pcbi.1007007
- Perkel, J. M. (2018). Why Jupyter is data scientists' computational notebook of choice. *Nature* 563(7729): 145–146. https://doi.org/10.1038/d41586-018-07196-1
- Kery, M. B., Radensky, M., Arya, M., John, B. E., & Myers, B. A. (2018). The Story in the Notebook: Exploratory Data Science Using a Literate Programming Tool. *Proc. CHI 2018*. https://doi.org/10.1145/3173574.3173748

### B. Scientific-software documentation best practice

**Diátaxis** (Procida, https://diataxis.fr): four orthogonal genres — *tutorial* (learning-oriented, hands-on, instructor present in prose), *how-to guide* (problem-oriented, for competent users), *reference* (information-oriented), *explanation* (understanding-oriented). The single most common defect in scientific-Python docs is mixing them. An eegdash *tutorial* must put the learner on rails to a guaranteed successful concrete result; cross-link out to how-tos and reference rather than embedding them.

**Scikit-learn / sphinx-gallery conventions**
- Sphinx-Gallery generates HTML, Jupyter `.ipynb`, and downloadable `.py` from a single annotated `plot_*.py`. File naming convention: only `plot_*.py` is executed and screenshots captured; gallery folder needs a `GALLERY_HEADER.rst` (or `README.rst`) with an reST title and intro. Source: https://sphinx-gallery.github.io/stable/getting_started.html and https://sphinx-gallery.github.io/stable/syntax.html
- Each example begins with a module-level docstring whose first line is a reST title (`===`), followed by a 1–2-paragraph motivational introduction. Subsequent prose blocks are introduced with `# %%` or a line of `#`s. This produces clean HTML with thumbnail, downloadable notebook, and binder/launch links.
- The scikit-learn gallery (https://scikit-learn.org/stable/auto_examples/) is the canonical exemplar: each example has a clear research question, a one-paragraph context, ~30–150 lines of code with interleaved prose, 1–4 figures with axis labels and titles, and a "References" or "See also" footer linking back to the user guide.

**MNE-Python tutorials** (https://mne.tools/stable/auto_tutorials/index.html) are an ordered sequence — Introductory → Raw → Preprocessing → Epochs → Evoked → Time-Frequency → Source space → Stats — with tutorials assuming earlier ones; this sequenced ordering is itself a pedagogical choice borrowed by Braindecode (https://braindecode.org/stable/auto_examples/index.html).

**Reproducible-research / executable-paper / FAIR foundations**
- Wilson, G., et al. (2014). Best practices for scientific computing. *PLOS Biology* 12(1): e1001745. https://doi.org/10.1371/journal.pbio.1001745
- Wilson, G., Bryan, J., Cranston, K., Kitzes, J., Nederbragt, L., & Teal, T. K. (2017). Good enough practices in scientific computing. *PLOS Computational Biology* 13(6): e1005510. https://doi.org/10.1371/journal.pcbi.1005510
- Sandve, G. K., Nekrutenko, A., Taylor, J., & Hovig, E. (2013). Ten simple rules for reproducible computational research. *PLOS Computational Biology* 9(10): e1003285. https://doi.org/10.1371/journal.pcbi.1003285
- Osborne, J. M., et al. (2014). Ten Simple Rules for Effective Computational Research. *PLOS Computational Biology* 10(3): e1003506.
- Mensh, B., & Kording, K. (2017). Ten simple rules for structuring papers. *PLOS Computational Biology* 13(9): e1005619 — applies almost verbatim to tutorial structure (one central idea, a single narrative arc, figures lead the story).
- Wilkinson, M. D., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data* 3: 160018. https://doi.org/10.1038/sdata.2016.18

### C. Neuroscience / EEG education evidence

- Cisotto, G., & Chicco, D. (2024). Ten quick tips for clinical electroencephalographic (EEG) data acquisition and signal processing. *PeerJ Computer Science* 10: e2256. https://doi.org/10.7717/peerj-cs.2256 — covers reference choice, filtering, artefact removal, ICA, validation, sample-size, and reporting; an ideal "domain-correctness" checklist.
- Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience* 7: 267. https://doi.org/10.3389/fnins.2013.00267 — establishes the canonical MNE pipeline that any eegdash tutorial should integrate with.
- Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping* 38(11): 5391–5420. https://doi.org/10.1002/hbm.23730 — Braindecode's foundational paper.
- Pernet, C. R., et al. (2019). EEG-BIDS, an extension to the brain imaging data structure for electroencephalography. *Scientific Data* 6: 103. https://doi.org/10.1038/s41597-019-0104-8 — eegdash is BIDS-first, so any tutorial should make BIDS entities (`subject`, `task`, `session`, `run`) visible.
- van Viegen, T., et al. (Neuromatch Academy). (2021). Neuromatch Academy: Teaching computational neuroscience with global accessibility. *Trends in Cognitive Sciences* 25(7): 535–538. https://doi.org/10.1016/j.tics.2021.03.018 — validates pod-based, tutorial-driven, project-capstoned online instruction at scale (1,757 students; 95%+ satisfaction reported in associated *JOSE* publication).
- 't Hart, B. M., et al. (2022). Neuromatch Academy: a 3-week, online summer school in computational neuroscience. *Journal of Open Source Education* 5(49): 118. https://doi.org/10.21105/jose.00118 — the curriculum is openly licensed and is a high-quality model of cognitive-load-managed Jupyter tutorials at https://compneuro.neuromatch.io/.
- Millman, K. J., Brett, M., Barnowski, R., & Poline, J.-B. (2018). Teaching computational reproducibility for neuroimaging. *Frontiers in Neuroscience* 12: 727. https://doi.org/10.3389/fnins.2018.00727 — argues for project-based, code-and-stats-first neuroimaging teaching.

### D. Engagement and motivation

- Ryan, R. M., & Deci, E. L. (2000). Self-determination theory and the facilitation of intrinsic motivation, social development, and well-being. *American Psychologist* 55(1): 68–78. https://doi.org/10.1037/0003-066X.55.1.68 — autonomy, competence, relatedness.
- Ryan, R. M., & Deci, E. L. (2020). Intrinsic and extrinsic motivation from a self-determination theory perspective. *Contemporary Educational Psychology* 61: 101860.
- Hochanadel, A., & Finamore, D. (2015). Fixed and growth mindset in education and how grit helps students persist in the face of adversity. *Journal of International Education Research* 11(1): 47–50.
- Carlson, J. R., et al. (2018). *Getting messy with authentic data: Exploring the potential of using data from scientific research to support student data literacy*. *CBE—Life Sciences Education* 17(1): es1. https://doi.org/10.1187/cbe.18-02-0023 — empirical case for authentic, "messy" data in pedagogy.

**Implication:** The opening of every tutorial should answer "Why would a researcher care?" with a real neuroscience question on real data (eegdash gives access to OpenNeuro/HBN/NEMAR datasets — use them). This serves SDT's *competence* (the learner achieves a real result) and *autonomy* (they can swap their own subject IDs).

---

### E. Concrete validation rubric for the AI agent's output

Use this as a PR-review checklist. Each item is a yes/no/partial; aim for **≥ 90% Yes** before merging.

#### E.1 Structural / scikit-learn-and-sphinx-gallery conformity
1. File is named `plot_<short_descriptive_name>.py` (so sphinx-gallery executes it and captures figures).
2. First lines are a triple-quoted module docstring whose first line is a reST H1 title (e.g., `===` underline) and whose second paragraph is a 2–4-sentence motivating problem statement that names the dataset and the scientific question.
3. The example sits in a gallery sub-folder with a `GALLERY_HEADER.rst` (or `README.rst`) defining the section.
4. Code blocks are separated from prose using consistent block delimiters (`# %%` or a line of 79 `#`s — pick one and stick to it).
5. Imports are grouped (stdlib → third-party → eegdash/braindecode/mne) at the top, with no hidden imports later.
6. Every figure has axis labels, units, a title, and a legend if multiple traces.
7. The example ends with a "References" / "See also" / "Next steps" block linking to the user guide, related examples, and primary literature (DOI links).
8. Total runtime on a CPU is reasonable (target < ~3 minutes for a "core" example; longer ones must be marked and use `mini=True` HBN releases or a single subject).
9. The example is fully runnable top-to-bottom from a clean kernel (Rule 4 of Rule et al. 2019).
10. There is a downloadable `.ipynb` and `.py` link generated by sphinx-gallery (default behaviour — verify the build).

#### E.2 Pedagogical (cognitive-load + PRIMM + scaffolding)
11. The notebook is **one narrative** with a beginning ("Why we care"), middle ("How we do it"), end ("What we found and what's next") (Rule et al. 2019, Rule 1; Mensh & Kording 2017).
12. Worked-example structure for novices: complete code is shown, then explained, before the learner is asked to modify (Sweller & Cooper 1985).
13. **PRIMM** elements are visible: at least one **Predict** prompt before a code cell ("What do you think `dataset.description` will contain?"), the **Run** cell, an **Investigate** prose paragraph that explains the output, an explicit **Modify** exercise (e.g., "change `task='RestingState'` to `'contrastChangeDetection'`"), and a closing **Make** mini-challenge (Sentance & Waite 2017).
14. Cognitive load is managed: each cell does **one** conceptual thing; long expressions are broken into named intermediate variables; figures and the prose that explains them are adjacent (no split-attention).
15. The tutorial respects the **expertise-reversal effect**: it is tagged for level (beginner / intermediate / advanced) and does not over-explain at the wrong level (Kalyuga et al. 2003). Beginner tutorials use full worked examples; advanced ones can use Parsons-style or faded blanks.
16. There is at least one **faded** or **completion** exercise (Weinman et al. 2021) — e.g., "Fill in the band-pass cutoffs in this preprocessing pipeline" — to bridge reading-to-writing.
17. Errors are shown intentionally at least once and recovered from in prose ("If you forget `cache_dir=`, you'll see this error — here's why", per Nederbragt et al. 2020).
18. Active-learning prompts appear every ~10–15 minutes of estimated reading (Brown & Wilson 2018, Tip 6).
19. Vocabulary is introduced before it is used (BIDS, montage, epoch, ICA, ERP, etc.) with one-sentence operational definitions.
20. The opening sets explicit **learning objectives** (3–5 bullets: "After this tutorial you will be able to …").

#### E.3 Technical (reproducibility, correctness, runnability)
21. All RNGs are seeded (`np.random.seed`, `torch.manual_seed`, sklearn `random_state=`).
22. Versions are pinned or printed (`%pip list` cell or `mne.sys_info()` / `print(eegdash.__version__)`).
23. The example uses a **small, deterministic subset** of data (e.g., one subject, one task, `mini=True` for HBN) so it runs in CI.
24. Data are downloaded via `eegdash`'s caching API (no manual paths) and the cache directory is parametrised, not hard-coded.
25. The notebook restarts-and-runs-all without warnings other than expected scientific ones (e.g., MNE filter-design info).
26. Outputs (figures, prints) are committed as artefacts (sphinx-gallery captures them automatically — verify the rendered HTML).
27. Computation is split so a learner can stop at any cell with a meaningful intermediate result (Rule et al. 2019, Rule 7).
28. The example handles the offline/airgapped case (eegdash has `download=False`) and documents network requirements.
29. Heavy training loops use a tiny number of epochs / a small model and clearly say "increase this for real work" — Braindecode style.
30. The notebook does not silently `pip install` packages mid-execution.

#### E.4 Engagement (narrative, authentic data, motivation)
31. The first 5 lines name a real neuroscience question (e.g., "Can we tell from a 2-second EEG snippet whether a child's eyes are open or closed?") rather than a generic "this example shows how to use class X."
32. The dataset is a **real, citable** EEG-DaSh / OpenNeuro dataset with DOI (eegdash auto-generates citations — surface them).
33. The result has scientific meaning (chance vs above-chance, an interpretable spectrum, an ERP that resembles the literature) — not just "the code runs."
34. The conclusion includes a "Try it yourself" / "Extensions" section that gives the learner autonomy (SDT) — at least 3 suggested modifications of varying difficulty.
35. Tone is "we"-inclusive, present-tense, and explains the *why* of choices (e.g., "we resample to 100 Hz to keep this example fast; for real work, 250 Hz is typical").
36. There are **figures the learner can recognise** from the EEG textbook canon (a topomap, a PSD, an ERP butterfly plot, a confusion matrix), so the work feels like real EEG research.

#### E.5 Domain-correctness (EEG / neuroscience accuracy)
37. Filtering choices are justified and use causal/non-causal flags appropriately; pass-band, stop-band, and filter type are reported (Cisotto & Chicco 2024, Tips 4–5).
38. Reference scheme and montage are explicit (`set_montage`, `set_eeg_reference`).
39. Bad-channel handling is at least mentioned, even if skipped for simplicity.
40. Artefact strategy (ICA, autoreject, simple thresholding) is named and a citation is given.
41. Epochs use a baseline correction window, and the sign convention/units are stated (µV).
42. If a classifier is trained, train/test split is **subject-aware** (no leakage across subjects/sessions) — this is the single most common EEG-ML mistake.
43. Class balance and chance level are explicitly computed and displayed alongside accuracy.
44. BIDS entities used in the query (`dataset`, `subject`, `task`, `session`, `run`) are surfaced in prose, not hidden.
45. The tutorial cites the **dataset paper** (DOI), the **eegdash** entry, **MNE-Python** (Gramfort et al. 2013), and **Braindecode** (Schirrmeister et al. 2017) where appropriate.
46. The tutorial does not over-claim: phrasing is hedged ("on this single subject we observe…") and limitations are flagged.

#### E.6 Diátaxis purity
47. The document is unambiguously a *tutorial* (one rail, one outcome) — reference material lives in API docs, not embedded.
48. Where deeper conceptual material would help, the tutorial *links out* to an explanation page rather than inlining it.
49. Where a competent user would want a quick recipe, it is split into a separate how-to.

---

### F. Recommended tutorial structure for `eegdash`

Given that `eegdash` already mirrors the sphinx-gallery layout (`generated/auto_examples/`), I recommend a **three-tier ladder** with one canonical "first tutorial" plus topical galleries.

#### F.1 Canonical "first tutorial" — `plot_quickstart_resting_state.py`
A 60–90-minute, single-subject, single-task notebook that exercises the full library in a meaningful way. Suggested structure:

1. **Title + 3-sentence motivation** (eyes-open vs eyes-closed alpha rhythm — a textbook neuroscience finding everyone can verify).
2. **Learning objectives** (5 bullets).
3. **Setup** — one cell that imports, prints versions, sets seeds and the cache dir.
4. **PRIMM-Predict #1**: "What do you think the database contains for `ds002718`?"
5. **Discover the data** — `EEGDash().find(...)` returning records, with prose explaining BIDS entities. Print the first record as a dict.
6. **Load one subject** — `EEGDashDataset(...)` worked example.
7. **PRIMM-Investigate**: a small `Predict → Run → Investigate` block on `len(dataset)`, channel names, sampling frequency.
8. **Preprocess via Braindecode** — band-pass, resample, average reference. Justify each choice in prose with a Cisotto & Chicco citation.
9. **Visualise** — raw plot, PSD, topomap. Show the alpha bump for eyes-closed.
10. **PRIMM-Modify**: "Change the band-pass to 1–8 Hz; what happens to the alpha bump?"
11. **A minimal classifier** — logistic regression on band-power features (keep it simple; no deep nets in the first tutorial). Subject-aware split, chance-level baseline, confusion matrix.
12. **PRIMM-Make**: "Repeat the analysis on another subject of your choice and compare."
13. **Caveats**, **Citations** (dataset DOI, eegdash, MNE-Python, Braindecode), **See also** (links to deeper tutorials).

#### F.2 Topical sub-galleries (mirror MNE-Python's structure)
- **`auto_examples/core/`** — discovering, querying, caching data; offline mode; metadata.
- **`auto_examples/preprocessing/`** — filtering, ICA, bad-channel handling, BIDS round-tripping with `mne-bids`.
- **`auto_examples/features/`** — band-power, connectivity, time-frequency.
- **`auto_examples/decoding/`** — linear models, then EEGNet/Braindecode CNNs (mirrors Braindecode's `model_building/` examples).
- **`auto_examples/eeg2025/`** — challenge tutorials (already exists; keep these as the "advanced" tier).
- **`auto_examples/scaling/`** — multi-subject, group analysis, HPC patterns. Includes the existing offline tutorial.

Each sub-gallery has its own `GALLERY_HEADER.rst` with a 1-paragraph orientation and a difficulty tag.

#### F.3 Cross-cutting conventions
- A standardised header banner in each example: difficulty (★/★★/★★★), estimated runtime, hardware (CPU/GPU), dataset size, prerequisite tutorials.
- Every example ends with a "What you learned" recap plus 3 graded "Extensions."
- A top-level **"How to read this gallery"** explanation page (Diátaxis explanation) describing the difficulty ladder and the recommended order.
- Use the same colour palette and figure style across the gallery (a small `eegdash/_plot_style.py` helper).
- Ensure CI (GitHub Actions) builds the gallery on every PR and fails on warnings — this enforces reproducibility automatically.

---

## Caveats

- **Specific empirical effect sizes** for Software Carpentry and Neuromatch (e.g., the "130% improvement" / "85% satisfaction" figures) come from self-report and short pre/post tests in non-randomised designs; they show consistent direction but should not be cited as causal estimates. The Carpentries' own write-up (Wilson 2016, F1000) is candid about this limitation.
- **Live-coding evidence is mixed.** Rubin (2013) was positive; the more recent Shah et al. (2023, ICER/ITiCSE) controlled study found minimal effects on grades and slightly worse exam performance with live coding, even though process behaviours improved. The safe synthesis is the Nederbragt et al. (2020) "ten quick tips" — adopt the *narrating-while-coding* style in tutorial prose without over-claiming pedagogical superiority.
- **PRIMM was developed for K-12 / introductory CS**, not for adult researchers learning a domain library. Its usefulness as a *structuring template* (Predict/Run/Investigate/Modify/Make) generalises well, but the empirical effect sizes do not directly transfer to PyData tutorials.
- **The "Diátaxis is empirical" claim** circulating online is overstated; Procida's framework is a craft/reflective synthesis, not a controlled trial. It is widely adopted (Django, Cloudflare, Gatsby, Canonical, NumPy) because it works, not because of an RCT.
- **`eegdash` is young (v0.4.x as of 2026).** The API names cited here (`EEGDash`, `EEGDashDataset`, `EEGChallengeDataset`, `RELEASE_TO_OPENNEURO_DATASET_MAP`) are taken from current docs at https://eegdash.org and may evolve; the rubric should be applied against the version pinned in the tutorial.
- **Some references are PLOS "Ten Simple Rules" / "Quick Tips" articles** which are editorial syntheses, not primary research; treat them as expert distillations of the underlying empirical literature (which they cite).
- **The expertise-reversal effect implies you should produce *parallel* novice and expert versions** of key tutorials, not a single one-size-fits-all document. This is the main weakness of most current scientific-Python galleries (including scikit-learn's), and a place where eegdash can innovate with explicit "★/★★/★★★" tagging.
- **Cognitive Load Theory measurements** (subjective NASA-TLX, dual-task) are rarely applied to written tutorials in the wild; the rubric in §E uses CLT as a *design heuristic*, not as a measurement instrument. If you want to validate empirically, you can pilot the tutorial with 5–10 representative learners, ask them to think aloud, and time-to-completion + error-recovery counts will catch most cognitive-load failures more cheaply than instrumented measures.