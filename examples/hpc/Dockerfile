
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libopenblas-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install \
      eegdash \
      braindecode \
      mne mne-bids \
      scipy scikit-learn matplotlib pandas \
      torch torchaudio wfdb

WORKDIR /workspace
CMD ["python", "--version"]
