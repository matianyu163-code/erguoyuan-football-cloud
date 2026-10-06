# PHASE 3 COMPLETION REPORT

## 1. 新增文件

- `configs/models.yaml`
- `docs/phase3_base_models.md`
- `src/erguoyuan_football/models/{artifact,base,bivariate_poisson,comparison,config,dixon_coles,elo,hierarchical_bayes,penaltyblog_adapter,pi_rating,rating_probability,registry,run_base_models,runner,score_matrix,training}.py`
- `src/erguoyuan_football/backtesting/{__init__,base_model_backtest}.py`
- `src/erguoyuan_football/markets/{__init__,from_score_matrix}.py`
- `tests/unit/test_phase3_models.py`
- `tests/unit/test_phase3_backtest.py`
- `tests/integration/test_phase3_models.py`
- `THIRD_PARTY_NOTICES.md`

## 2. 修改文件

- `AGENTS.md`, `README.md`, `pyproject.toml`
- `docs/audit_contract.md`, `docs/v51_output_contract.md`
- Existing Phase 2 contracts, fixture schema, model package exports and static-formatting fixes in `contracts/`, `data/`, `input/`, `output/v51/`, and their existing tests.

## 3. 实际安装第三方依赖

Python 3.11.16 environment:

| package | version |
|---|---:|
| duckdb | 1.5.6 |
| pydantic | 2.13.5 |
| pytz | 2026.4 |
| penaltyblog | 1.12.2 |
| numpy | 2.3.5 |
| pandas | 3.0.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| PyYAML | 6.0.3 |
| pytest | 8.4.2 |
| ruff | 0.16.9 |
| mypy | 2.3.1 |

PyMC and ArviZ were not installed because current releases investigated for this Python 3.11 project require Python 3.12. Hierarchical Bayes V1 uses penaltyblog's public MCMC API instead.

## 4. LICENSE 检查

`penaltyblog` 1.12.2 is MIT licensed and the installed distribution includes its `licenses/LICENCE` file. NumPy, pandas, SciPy, scikit-learn, joblib and DuckDB are recorded as BSD or bundled permissive licenses in `THIRD_PARTY_NOTICES.md`; PyYAML is MIT. No third-party source was copied into `src/`.

## 5. 五个模型状态

| model | status | implementation |
|---|---|---|
| Dixon–Coles | IMPLEMENTED | penaltyblog adapter |
| Bivariate Poisson | IMPLEMENTED | penaltyblog adapter with fitted shared `lambda3` |
| Bayesian Hierarchical | IMPLEMENTED | penaltyblog public MCMC adapter with diagnostics and posterior intervals when exposed |
| Elo | IMPLEMENTED | project implementation plus separately fitted multinomial mapper |
| Pi Rating | IMPLEMENTED | project implementation with separate home/away strengths plus mapper |

All five declare `REAL_IMPLEMENTATION`. A model with insufficient data returns `UNAVAILABLE`; a model with failed numerical or sampling execution returns `FAILED` with a reason.

## 6. 统一接口

`BaseFootballModel` provides fit, predict, predict_many, save, load, metadata and PIT validation. `ModelPrediction` carries the Phase 2 audit fields plus expected goals, correlation, execution time, warnings and metadata. `ModelArtifact` stores lineage hashes, cutoff and dependency versions. `ModelRegistry` only instantiates; `ModelRunner` checks requirements and isolates failures.

## 7. Score Matrix

Goal models return a square home-goals-by-away-goals matrix with inclusive configurable `max_goals`. Truncated mass is conditionally renormalized and recorded as `retained_mass`, `tail_mass` and `tail_policy`. 1X2, double chance, BTTS, totals and exact scores are derived from the same matrix.

## 8. Walk-forward

`backtesting/base_model_backtest.py` creates expanding chronological windows, skips windows whose result availability arrives after the next kickoff, and evaluates only `is_oos=True` predictions. It reports Log Loss, multiclass Brier, RPS, Accuracy, sample size, competition, date range and model version.

## 9. 测试结果

Actual command:

```text
.\.venv\Scripts\python.exe -m pytest -q
156 passed in 26.62s
```

`ruff check src tests`: passed. Targeted new-engine command
`mypy src/erguoyuan_football/models src/erguoyuan_football/backtesting src/erguoyuan_football/markets`: passed with no issues in 22 files.

## 10. Integration result

The integration fixture is explicitly labelled `SYNTHETIC_TEST` and requires `allow_test_data=True`. It executes all five algorithms without probability mocks; the development Bayesian configuration passed diagnostics in the full five-model run, and the comparison report preserved each model's real status and metadata. This is a code-path test, not evidence of production league accuracy.

## 11. 数据泄漏

No leakage was found by the PIT, future-row rejection, pre-match rating, walk-forward late-evidence and OOS-only tests. Training rows require kickoff/result availability before `trained_until`; prediction requires `trained_until <= prediction_time < kickoff_time`; target matches in training are rejected.

## 12. Benchmark

League-frequency and independent-Poisson benchmark implementations and metric tests are present. No production benchmark score is reported because this checkout contains no authorized real historical league dataset; synthetic benchmark fixtures are never presented as production performance.

## 13. 尚未实现内容

At the time of the Phase 3 acceptance report, Dynamic Bayesian Poisson and the other listed modules remained out of scope. Phase 4 status is tracked separately in `phase4_dynamic_bayes_report.md`. SPI-like, Opta-like xG-Elo, Opta-like Supercomputer simulation, historical+market Bayesian Poisson, XGBoost, CatBoost, META STACKING, calibration, market/external-data fusion, context/lineup gates, Monte Carlo production simulation, betting selection and funds allocation remain out of scope. V5.1 remains frozen and `NOT_READY_FOR_PRODUCTION`.

The full legacy command `mypy src` still reports 79 pre-existing Phase 2 typing errors in the data/input/V5.1 scaffolding; Phase 3 modules themselves pass the targeted mypy command above. No Phase 3 model code is hidden behind that limitation.
