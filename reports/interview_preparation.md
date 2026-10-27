# Interview preparation

1. **Why did persistence win?** It has the lowest pooled CPU MAE across matched
   24-step folds; Ridge improves RMSE but misses the primary MAE criterion.
   Complexity is justified by measured benefit, not model branding.
2. **How did you prevent leakage?** The loader drops holdout labels; downstream
   prediction accepts training targets and future timestamps only. Per-fold
   lags and centering use past data, and whole-trace metadata is excluded.
3. **Why is active-count error zero?** The dataset selects complete VM traces.
   Persistence learns the last count; it does not hard-code 10,000. This does
   not validate production churn prediction, and relative MAE is undefined.
4. **Why use these folds?** Expanding origins 50/62/74/86 retain enough history
   while repeatedly testing the operational two-hour horizon. Overlap improves
   temporal coverage but creates dependent errors and limits confidence.
5. **What is a genuine recursive forecast?** Later predictions consume previous
   predictions, never validation actuals. A collection of one-step predictions
   supplied with observed future lags would exaggerate deployable accuracy.
6. **What does 9% fleet CPU mean operationally?** It is an unweighted VM average.
   It cannot establish consumed cores or safe host consolidation; high-CPU VMs
   and latency-sensitive workloads can be saturated at the same time.
7. **Are the uncertainty bounds calibrated?** No. They pool historical absolute
   residuals from few overlapping origins. Earlier-label coverage is descriptive
   and model selection still shares the same folds; longer independent history
   is needed to assess coverage and horizon-specific risk.
8. **How are missing readings treated?** A missing VM row/null average is excluded
   from both targets rather than filled with zero. Entire timestamp gaps fail.
   Apparent inactivity can therefore mean telemetry loss, requiring monitoring.
9. **Why avoid seasonal/deep models?** Only 110 training timestamps exist and
   timestamps are relative seconds. No daily/weekly cycle or data-intensive
   model is supported by the observation window.
10. **What would change in production?** Obtain longer representative history,
    exact capacities and placement, lifecycle events, queues, latency SLOs and
    scaling costs. Reserve independent chronological tests and assess policies,
    segment errors, drift, rare-event risk, cooldowns and rollback behavior.
