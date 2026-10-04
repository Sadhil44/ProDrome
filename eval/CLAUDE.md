# eval/ module rules

This directory belongs to the **collect** seat (Shaurya): the evaluation harness and the plots. It
is where the project's claims are either earned or exposed, so the rules here are stricter than
elsewhere.

- Every claim ships with the control-arm comparison. Kubernetes already self-heals; "we recovered
  the service" means nothing without "and stock Kubernetes didn't." A number with no control arm is
  not reportable, however good it looks.
- Report per fault type, never one aggregate. An average hides the cases where the method does not
  work, and those are the most informative part of the result.
- Build and report the simple baseline first — a threshold, a shallow rule. It is how anyone knows
  the complicated thing earned its place. If the simple version ties, we ship it and say so.
- State the provenance of every number in the same place as the number: which run ids, synthetic
  fixture or real cluster data, and how train and test were split. A figure whose provenance lives
  only in someone's head is not a result.
- Classifying injected faults is close to trivial — a CPU stressor spikes CPU, a memory stressor
  spikes memory — so a high classification accuracy proves little on its own. The lead-time curve and
  the control-arm comparison are the results that mean something; weight the report accordingly.
- Say what it cannot do, in the report itself: instant failures get no warning, no cross-service root
  cause, closed label set. Stating these is what makes everything else credible.
- Everything regenerates from a fresh clone with a committed command. A plot that cannot be
  regenerated is a picture, not evidence.
- Coordination happens on the CCP forum (MCP `ccp-forum`, session `prodrome`): set status, post
  results to `collect/updates`, publish the harness contract (which metrics, per fault type, how the
  control arm is run) in `collect/interfaces`, and put any headline number in `crosstalk/decisions`
  with its baseline beside it. See `forum/HANDOFF.md`.
