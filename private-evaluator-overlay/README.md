Development-only evaluator overlay for local benchmark runs.

This overlay lets the public execution_patch tasks run end-to-end on machines that do
not have the separate hidden-evaluator bundle. Each JSON file maps a public execution
task to its checked-in visible test selector.

Use this for local debugging and workflow verification only. It is not a claim-safe
substitute for the real hidden evaluators.
