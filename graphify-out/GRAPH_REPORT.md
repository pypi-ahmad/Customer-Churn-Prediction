# Graph Report - Customer-Churn-Prediction  (2026-09-23)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 105 nodes · 218 edges · 7 communities (6 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5ac0a262`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6

## God Nodes (most connected - your core abstractions)
1. `main()` - 15 edges
2. `generate_predictions()` - 12 edges
3. `main()` - 10 edges
4. `fit_and_predict_tabfm()` - 10 edges
5. `build_tabfm_classifier()` - 8 edges
6. `predict_mitra()` - 8 edges
7. `train_mitra()` - 8 edges
8. `prepare_split()` - 7 edges
9. `load_mitra()` - 7 edges
10. `predict_tabfm()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `generate_predictions()` --calls--> `load_mitra()`  [EXTRACTED]
  app.py → foundation_models.py
- `generate_predictions()` --calls--> `load_tabfm_from_context()`  [EXTRACTED]
  app.py → foundation_models.py
- `generate_predictions()` --calls--> `predict_mitra()`  [EXTRACTED]
  app.py → foundation_models.py
- `generate_predictions()` --calls--> `predict_tabfm()`  [EXTRACTED]
  app.py → foundation_models.py
- `main()` --calls--> `release_cuda()`  [EXTRACTED]
  train.py → foundation_models.py

## Import Cycles
- None detected.

## Communities (7 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.14
Nodes (25): configure_logging(), ensure_streamlit_config(), generate_predictions(), load_data(), load_model_bundle(), main(), preprocess_for_inference(), Any (+17 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (19): argparse, flaml, download_tabfm_checkpoint(), gc, imblearn_over_sampling, joblib, lazypredict_supervised, logging (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (17): ref_node_child_process, ref_node_fs, ref_node_path, add(), batches, edges, fileNodes, graph (+9 more)

### Community 3 - "Community 3"
Cohesion: 0.18
Nodes (16): raw_features_for_inference(), AutoML, numpy, pytest, test_metrics_use_positive_churn_class(), test_prepare_split_excludes_target_and_identifiers(), test_raw_feature_validation_reports_missing_columns(), test_target_encoding_marks_attrition_as_positive() (+8 more)

### Community 4 - "Community 4"
Cohesion: 0.34
Nodes (15): build_tabfm_classifier(), fit_and_predict_tabfm(), load_mitra(), load_tabfm_from_context(), predict_mitra(), predict_tabfm(), Any, DataFrame (+7 more)

### Community 5 - "Community 5"
Cohesion: 0.33
Nodes (6): build_model_factory(), load_data(), Any, DataFrame, Path, run_lazypredict()

## Knowledge Gaps
- **14 isolated node(s):** `batches`, `edges`, `fileNodes`, `graph`, `ids` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 41 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 4` to `Community 0`, `Community 1`, `Community 3`, `Community 5`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `generate_predictions()` connect `Community 0` to `Community 3`, `Community 4`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **What connects `batches`, `edges`, `fileNodes` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.13846153846153847 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.11904761904761904 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.10526315789473684 - nodes in this community are weakly interconnected._