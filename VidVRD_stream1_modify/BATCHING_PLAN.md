# Cached inference batching

1. Freeze the current serial stream implementation in tests/reference_serial.py.
2. Add a reusable batch step: validate all events before mutation, encode current
   segments once, update per-pair raw spatial caches, right-pad history windows,
   classify once, transfer the complete score matrix once, associate in input order.
3. Index cache filenames without opening feature files. Load at most the configured
   active videos. Schedule only one completion-time group per video per batch;
   do not advance a video until the yielded batch has been processed. Flush EOF
   after processing its last batch, then release input and GPU working state.
4. Integrate into the existing inference/validation replay, preserving logging-only
   output policy and in-memory event/final-result formats. Do not edit training logic,
   model parameters or association.
5. Test against the frozen reference, including sliding windows, variable lengths,
   prefix consistency, EOF/reset isolation, ties and lazy loading. Compare a small
   real checkpoint/data subset on available hardware without profiling.
6. Document independent inference controls and CPU/GPU verification limitations.
