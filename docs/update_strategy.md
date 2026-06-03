# Update Strategy

To keep the PIR database fresh without rebuilding the full dataset:

1) **Initial sync**: Client loads full `database.bin` once (local Reth extract or existing copy).
2) **Incremental Updates**:
   - A separate service monitors the chain for state changes.
   - It publishes a compact list of delta updates for each block (see `plinko/docs/delta-format.md`).
   - **Client Update**: The client updates local hints using XOR:
     `NewHint = OldHint XOR (OldVal at Index) XOR (NewVal at Index)`
   - This is O(1) per changed state entry.
