<a id="data-summary"></a>

# Datasets Catalog

EEG-DaSh is a data-sharing archive for MEEG (EEG, MEG) recordings contributed by collaborating labs. It preserves publicly funded research data and exposes it in a form that machine learning and deep learning workflows can use directly.

<svg version="1.1" width="2.0em" height="2.0em" class="sd-octicon sd-octicon-database sd-text-primary" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 1.25c2.487 0 4.773.402 6.466 1.079.844.337 1.577.758 2.112 1.264.536.507.922 1.151.922 1.907v12.987l-.026.013h.026c0 .756-.386 1.4-.922 1.907-.535.506-1.268.927-2.112 1.264-1.693.677-3.979 1.079-6.466 1.079s-4.774-.402-6.466-1.079c-.844-.337-1.577-.758-2.112-1.264C2.886 19.9 2.5 19.256 2.5 18.5h.026l-.026-.013V5.5c0-.756.386-1.4.922-1.907.535-.506 1.268-.927 2.112-1.264C7.226 1.652 9.513 1.25 12 1.25ZM4 14.371v4.116l-.013.013H4c0 .211.103.487.453.817.351.332.898.666 1.638.962 1.475.589 3.564.971 5.909.971 2.345 0 4.434-.381 5.909-.971.739-.296 1.288-.63 1.638-.962.349-.33.453-.607.453-.817h.013L20 18.487v-4.116a7.85 7.85 0 0 1-1.534.8c-1.693.677-3.979 1.079-6.466 1.079s-4.774-.402-6.466-1.079a7.843 7.843 0 0 1-1.534-.8ZM20 12V7.871a7.85 7.85 0 0 1-1.534.8C16.773 9.348 14.487 9.75 12 9.75s-4.774-.402-6.466-1.079A7.85 7.85 0 0 1 4 7.871V12c0 .21.104.487.453.817.35.332.899.666 1.638.961 1.475.59 3.564.972 5.909.972 2.345 0 4.434-.382 5.909-.972.74-.295 1.287-.629 1.638-.96.35-.33.453-.607.453-.818ZM4 5.5c0 .211.103.487.453.817.351.332.898.666 1.638.962 1.475.589 3.564.971 5.909.971 2.345 0 4.434-.381 5.909-.971.739-.296 1.288-.63 1.638-.962.349-.33.453-.607.453-.817 0-.211-.103-.487-.453-.817-.351-.332-.898-.666-1.638-.962-1.475-.589-3.564-.971-5.909-.971-2.345 0-4.434.381-5.909.971-.739.296-1.288.63-1.638.962C4.104 5.013 4 5.29 4 5.5Z"></path></svg>

**824**

<svg version="1.1" width="2.0em" height="2.0em" class="sd-octicon sd-octicon-people sd-text-secondary" viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 8a5.5 5.5 0 1 1 8.596 4.547 9.005 9.005 0 0 1 5.9 8.18.751.751 0 0 1-1.5.045 7.5 7.5 0 0 0-14.993 0 .75.75 0 0 1-1.499-.044 9.005 9.005 0 0 1 5.9-8.181A5.496 5.496 0 0 1 3.5 8ZM9 4a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm8.29 4c-.148 0-.292.01-.434.03a.75.75 0 1 1-.212-1.484 4.53 4.53 0 0 1 3.38 8.097 6.69 6.69 0 0 1 3.956 6.107.75.75 0 0 1-1.5 0 5.193 5.193 0 0 0-3.696-4.972l-.534-.16v-1.676l.41-.209A3.03 3.03 0 0 0 17.29 8Z"></path></svg>

**46,374**

<svg version="1.1" width="2.0em" height="2.0em" class="sd-octicon sd-octicon-clock sd-text-info" viewBox="0 0 24 24" aria-hidden="true"><path d="M12.5 7.25a.75.75 0 0 0-1.5 0v5.5c0 .27.144.518.378.651l3.5 2a.75.75 0 0 0 .744-1.302L12.5 12.315V7.25Z"></path><path d="M12 1c6.075 0 11 4.925 11 11s-4.925 11-11 11S1 18.075 1 12 5.925 1 12 1ZM2.5 12a9.5 9.5 0 0 0 9.5 9.5 9.5 9.5 0 0 0 9.5-9.5A9.5 9.5 0 0 0 12 2.5 9.5 9.5 0 0 0 2.5 12Z"></path></svg>

**88,183**

<svg version="1.1" width="2.0em" height="2.0em" class="sd-octicon sd-octicon-pulse sd-text-success" viewBox="0 0 24 24" aria-hidden="true"><path d="M9.002 2.5a.75.75 0 0 1 .691.464l6.302 15.305 2.56-6.301a.75.75 0 0 1 .695-.468h4a.75.75 0 0 1 0 1.5h-3.495l-3.06 7.532a.75.75 0 0 1-1.389.004L8.997 5.21l-3.054 7.329A.75.75 0 0 1 5.25 13H.75a.75.75 0 0 1 0-1.5h4l3.558-8.538a.75.75 0 0 1 .694-.462Z"></path></svg>

**5**

Browse the catalog interactively at
[eegdash.org/dataset_summary.html](https://eegdash.org/dataset_summary.html).
The same data is available programmatically in Python:

```default
from eegdash import EEGDashDataset

# List every dataset
records = EEGDashDataset.list_datasets()

# Filter by task, modality, subject count, …
rest_datasets = EEGDashDataset.list_datasets(task="rest")
```

Or via the HTTP API at `https://data.eegdash.org`
(see the `/docs` Swagger UI and the
[api catalog](/.well-known/api-catalog)).

The archive is still in beta testing mode, so be kind.
