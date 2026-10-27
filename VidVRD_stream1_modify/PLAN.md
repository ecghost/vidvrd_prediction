# Stage 1 streaming implementation

1. Inspect source and cache contracts (completed): pair filename suffix is six digits;
   GT cache has pair-level duration and boxes, but no track IDs. Tail >=10 frames
   is retained; smaller tails are discarded by original extraction.
2. Preserve spatial parameter names/shapes; introduce causal/padding-aware temporal
   encoding, window-local positions, current-only supervision and valid object means.
3. Index training windows without copying overlapping arrays. Replay test segments
   by completion time, expose stream_step, cache only spatial representations.
4. Implement independent predicate top-N membership association, gap reset, explicit flush and immutable
   event snapshots. Reuse original evaluator only, never offline association.
5. Validate synthetic causal/padding/cache/gradient/association/prefix contracts,
   then document paths, commands, limitations and real-data verification status.

No original source, features or weights are modified. No training or downloads run
automatically. Fixed text vectors can be supplied offline; a local CLIP checkpoint
may optionally generate them using the original prompt templates.
