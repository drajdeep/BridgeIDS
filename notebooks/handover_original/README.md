Original notebooks received at hand-over, kept unchanged for traceability.
See `docs/PLAN_COVERAGE.md` §1 for the audit. In short: the ForgeIIoT notebook's 100 % accuracy comes from capture-order
leakage (`frame.number`, `frame.time_epoch`, `frame.time_relative`), and the NSL ARFF files carry only binary labels.
All results in BridgeIDS are produced by the leak-free pipeline in `src/bridgeids`.
