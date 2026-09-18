# topic_arm/ — the topic-model / ideology-recoverability arm (RQ1 of the progress report; not used by the paper précis)
nlp/      ad-span excision pipeline, LDA fits and K sweep, ridge/LOSO ideology harness, topic PCA and diagnostics, regime preprocessing, guest extraction and DIME guest matching, figure code
r/        R preprocessing regimes (preText, udpipe) and the installed library
figures/  fig1–fig7 for the topic-arm report
output/   remaining topic-arm outputs: the K=75 doc-topic matrix, 500-word chunk tables, LDA/embedding/PCA reports, ideology_targets_204, the DIME guest-matching audit trail and Stage-3 search batches
Large intermediates (sentence_embeddings_500.npy, adspan/, regimes/, per-K doc-topic files other than K=75) were deleted on 11 Sept 2026 and are regenerable from the scripts.
Running: the scripts import `step1_frame.config` and `nlp.*` as packages from the repo root. Run them as
    PYTHONPATH=topic_arm .venv/bin/python -m nlp.<script>
