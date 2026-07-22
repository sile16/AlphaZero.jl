# TensorBoard dashboard

Start TensorBoard with the session directory printed by `training_server.jl`, then
open the **Custom Scalars** tab for the curated 27-chart operator view. The standard
**Scalars** tab remains available for every diagnostic. Metric names use numbered,
stable namespaces so its cards are also grouped and ordered by purpose.
At a typical desktop width the curated view renders three charts per row, with
related series combined in one chart and categories in the order below.

The dashboard is organized by the question being answered:

1. **`01_ml_loss`** — Are total, policy, value, and invalid-action losses improving for
   both contact and race models?
2. **`02_ml_perf`** — Is the learner processing samples quickly, and are updates
   being skipped or becoming non-finite?
3. **`03_selfplay_perf`** — Are clients producing games, is MCTS fast, is neural
   inference efficiently batched, and are tree/bearoff lookups effective?
4. **`04_data`** — Is the replay buffer balanced, how old are the samples actually
   selected for training, and what fraction carries equity, chance-node, and
   exact-bearoff labels?
5. **`05_eval_strength`** — Is equity and win rate against the configured opponent
   improving, including side balance and value-prediction quality?
6. **`06_eval_bearoff`** — Are exact-table value error, policy top-k accuracy, raw
   NN versus MCTS choice quality, and regret improving?
7. **`07_system`** — Are CPU, GPU, and GPU-memory utilization healthy?
8. **`08_reliability`** — Are upload latency, retries, and rejected batches healthy?
9. **`09_promotion`** — Why were weights published or blocked?

The organization does not duplicate scalar data or run additional model
evaluations. The learner checks gradient finiteness before each optimizer update;
all other dashboard aggregation stays outside MCTS. Metrics that do not apply to a
run—for example PER metrics when PER is off—simply remain absent.

For a new session, the layout is immediately clean. When resuming an older event
directory, old scalar tags remain visible in the **Scalars** tab as historical data,
while new events use the current namespace. Start a new session directory for a
completely clean Scalars view.
