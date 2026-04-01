# Data Dictionary

All series in the `model_panel.parquet` output of `run_pipeline()`.

| Internal name | Raw Excel column | Description | Source | Unit (raw) | Frequency | Publication lag | Transform applied | Include in DFM |
|---|---|---|---|---|---|---|---|---|
| `consommation_ciment` | `Consommation_ciment` | Monthly cement consumption | APC / DEPF | tonnes (post-correction) | Monthly | ~35 days | Unit break fix → yoy_log_diff → standardize | ✅ Yes |
| `credits_equipement` | `credits_equipement` | Outstanding equipment credits | Bank Al-Maghrib | MDH | Monthly | ~35 days | yoy_log_diff → standardize | ✅ Yes |
| `credits_immobilier` | `credits_immobilier` | Outstanding real estate credits | Bank Al-Maghrib | MDH | Monthly | ~35 days | yoy_log_diff → standardize | ✅ Yes |
| `va_construction` | `VA CONSTRUCTION` | **Target variable** — Value Added, construction sector | HCP (CNT) | MDH current prices | Quarterly | ~90 days | quarter-end assignment only | ✅ Yes (target) |
| `ipai` | `L'IPAI` | Industrial Production Activity Index | HCP | Index base 100 | Quarterly | ~60 days | quarter-end assignment only | ✅ Yes |
| `lafarge_index` | `Indice_societes_construction_LAFARGEHOLCIM` | LafargeHolcim Maroc stock index | Casablanca Bourse | MAD/share | Monthly | Real-time | string fix → yoy_log_diff → standardize | ✅ Yes |
| `investissement_etat` | `Investissement_Etat` | State investment / budget execution | MEF / TGR | MDH | Monthly | ~20 days | monthly_diff | ❌ No (pending TGR confirmation) |
| `creation_emploi` | `Creation nette d emploi` | Net job creation in construction sector | HCP surveys | Count | Quarterly | ~90 days | quarter-end assignment only | ✅ Yes |

## Notes

- **`va_construction`** is the nowcast target. It appears in the panel with values only at quarter-end months (March, June, September, December). NaN in other months is intentional — the DFM's EM-Kalman algorithm treats it as partially observed.
- **`investissement_etat`** is excluded from the DFM (`include_in_model: false`) until the series definition is confirmed with TGR/MEF. See `configs/corrections.yaml` for details.
- **Cement break**: the correction factor of 759 is a **placeholder**. Confirm with APC before treating pipeline output as production-ready.
- All monthly series are transformed to year-over-year log differences (in percent) and then standardized (z-score) before entering the DFM.
- Quarterly series are NOT pre-interpolated; the DFM's aggregation constraint handles them.
