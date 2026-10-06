# Third-party notices — Phases 3–8

Versions below are the installed Python 3.11 environment observed through Phase 8. The project wraps public APIs; no third-party source code is copied into `src/`.

| Library | Installed version | License | Repository / docs | Use |
|---|---:|---|---|---|
| penaltyblog | 1.12.2 | MIT | https://github.com/martineastwood/penaltyblog | Dixon–Coles, Bivariate Poisson and Hierarchical Bayesian adapters |
| NumPy | 2.3.5 | BSD-3-Clause and bundled notices | https://github.com/numpy/numpy | matrices and numerical conversion |
| pandas | 3.0.6 | BSD-3-Clause | https://github.com/pandas-dev/pandas | penaltyblog runtime dependency |
| SciPy | 1.17.1 | BSD-3-Clause | https://github.com/scipy/scipy | probability distributions and penaltyblog runtime |
| scikit-learn | 1.9.1 | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn | rating-to-probability multinomial mapping |
| joblib | 1.6.0 | BSD-3-Clause | https://github.com/joblib/joblib | model artifact persistence |
| PyYAML | 6.0.3 | MIT | https://github.com/yaml/pyyaml | configuration loading |
| HTTPX | 0.28.1 installed in Phase 8 | BSD-3-Clause | https://github.com/encode/httpx | shared pooled network client |
| Beautiful Soup 4 | 4.15.0 | MIT | https://www.crummy.com/software/BeautifulSoup/ | parse allowlisted official fixture HTML |
| XGBoost | 3.2.0 | Apache-2.0 | https://github.com/dmlc/xgboost | official multiclass tree learner, behind `CoreXGBoostModel` |
| CatBoost | 1.2.10 | Apache-2.0 | https://github.com/catboost/catboost | official multiclass tree learner, behind `CoreCatBoostModel` |
| graphviz | 0.21 | MIT | https://github.com/xflr6/graphviz | transitive CatBoost dependency; not called by CORE |

`penaltyblog` ships its MIT license at `.venv/Lib/site-packages/penaltyblog-1.12.2.dist-info/licenses/LICENCE` and its repository identifies the project as MIT licensed. PyMC/ArviZ were investigated but not installed: current releases require Python 3.12 while this project is pinned to Python 3.11; hierarchical Bayesian V1 therefore uses penaltyblog's public MCMC wrapper.

Phase 6 adds no runtime dependency. Its numerical market fit and Bayesian MAP/Laplace routines use the already declared and installed SciPy, NumPy and scikit-learn dependencies; DuckDB remains the store. No market/OCR SDK was installed. Python runtime used for Phase 6 is 3.11.16.

HTTPX's upstream project metadata declares BSD-3-Clause. Version 0.28.1 was installed during Phase 8; injected transport tests pass. A live HTTPX request to the OpenFootball raw host failed because of the local TLS/proxy route, so a separately pinned Git checkout was used for offline import. No third-party model source is copied into CORE.

Phase 7 installed official Python 3.11 wheels for XGBoost 3.2.0 and CatBoost 1.2.10, plus CatBoost's graphviz dependency. Both learner projects declare Apache-2.0 in their PyPI metadata; graphviz includes an MIT license file in its installed distribution. CORE uses their public Python APIs and native save/load formats, without copying algorithm source.

Phase 8 source data: [OpenFootball football.json](https://github.com/openfootball/football.json), pinned commit `e6744429ee395bc86f247348c6184bb08d4eb361`, declares [CC0 1.0 Universal](https://github.com/openfootball/football.json/blob/master/LICENSE.md). Local raw files and their hashes are recorded in DuckDB; no dataset contents are copied into `src/`. The external API providers are unavailable without real access and are not represented as licensed historical coverage.

Phase 12 desktop build uses PyInstaller 6.22.3 (GPLv2-or-later with special exception; https://github.com/pyinstaller/pyinstaller) to bundle the Windows shell, and ReportLab 4.5.1 (BSD; https://www.reportlab.com/) to export PDF. Development performance measurement uses psutil 7.2.2 (BSD-3-Clause; https://github.com/giampaolo/psutil). Installed build dependencies are altgraph 0.17.5 (MIT), pefile 2024.8.26 (MIT), pywin32-ctypes 0.2.3 (BSD-3-Clause), and pyinstaller-hooks-contrib 2026.8 (Apache/GPL classified components; see its bundled LICENSE). Package metadata and bundled license files were checked in the Python 3.11 environment. No source code from these packages was copied into `src/`.

The Windows bundle also explicitly includes the existing runtime dependency pytz 2026.4 (MIT; https://pypi.org/project/pytz/) because DuckDB dynamically imports it for timezone handling. This inclusion was confirmed by running a packaged real development report, not only a GUI smoke test.

The packaged ISO region display-name aliases in `knowledge/entities/country_aliases.json` are data generated from Node.js ICU 78.3 `Intl.DisplayNames` (`en` and `zh-CN`) and .NET `RegionInfo` ISO alpha-3 codes. ICU embeds Unicode CLDR locale data; CLDR data is distributed under the Unicode-DFS-2016 license (https://cldr.unicode.org/index/downloads). These names and codes identify countries only; they do not assert football-provider verification or fixture availability.
