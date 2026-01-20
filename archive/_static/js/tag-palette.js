(function () {
  const valueClassMap = {
    categorization: {
      "convolution": "tag-conv",
      "recurrent": "tag-recurrent",
      "small-attention": "tag-smallattn",
      "filterbank": "tag-filterbank",
      "interpretability": "tag-interp",
      "spd": "tag-spd",
      "riemannian": "tag-spd",
      "large-brain-model": "tag-lbm",
      "graph-neural-network": "tag-gnn",
      "channel": "tag-channel",
    },
    "dataset-pathology": {
      "healthy": "tag-pathology-healthy",
      "clinical": "tag-pathology-clinical",
      "neurodevelopmental": "tag-pathology-neurodevelopmental",
      "neurological": "tag-pathology-neurological",
      "sleep": "tag-pathology-sleep",
      "__default": "tag-pathology-generic",
    },
    "dataset-modality": {
      "visual": "tag-modality-visual",
      "auditory": "tag-modality-auditory",
      "somatosensory": "tag-modality-somatosensory",
      "multisensory": "tag-modality-multisensory",
      "motor": "tag-modality-motor",
      "rest": "tag-modality-rest",
      "__default": "tag-modality-generic",
    },
    "dataset-type": {
      "perception": "tag-type-perception",
      "decision-making": "tag-type-decision",
      "rest": "tag-type-rest",
      "resting-state": "tag-type-rest",
      "sleep": "tag-type-sleep",
      "cognitive": "tag-type-cognitive",
      "clinical": "tag-type-clinical",
      "__default": "tag-type-generic",
    },
  };

  const knownValueClasses = Array.from(
    new Set(
      Object.values(valueClassMap)
        .map((mapping) => Object.values(mapping))
        .flat()
        .filter((cls) => typeof cls === "string")
    )
  );

  function apply(container) {
    const root = container || document;
    const tags = root.querySelectorAll('.tag[data-tag-kind]');
    tags.forEach((tag) => {
      const kind = tag.getAttribute('data-tag-kind');
      if (!kind) return;

      // ensure base kind class is present
      tag.classList.add(`tag-kind-${kind}`);

      const mapping = valueClassMap[kind] || {};
      const slug = tag.getAttribute('data-tag-value') || '';
      tag.setAttribute('data-tag-slug', slug);

      knownValueClasses.forEach((cls) => tag.classList.remove(cls));

      const assignment = mapping[slug] || mapping.__default;
      if (!assignment) {
        return;
      }

      if (Array.isArray(assignment)) {
        assignment.forEach((cls) => {
          if (cls) tag.classList.add(cls);
        });
      } else {
        tag.classList.add(assignment);
      }
    });
  }

  window.EEGDashTagPalette = window.EEGDashTagPalette || {};
  window.EEGDashTagPalette.apply = apply;

  document.addEventListener('DOMContentLoaded', function () {
    apply(document);
  });
})();
