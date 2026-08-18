# XAJ-SLW Model (Songliao)

`xaj_slw` is a XinAnJiang variant developed for the Songliao basin, using SMS3 and LAG3 routing (ported from hydromodeljava). It is the only registered model whose internal routing works in discharge (m^3/s); `UnifiedSimulator` converts its `qsim` back to runoff depth (mm) so results are comparable with other models and observations.

- Registered name: `xaj_slw`
- Parameters: 26
- Requires: `basin_area` (km^2) and `time_interval_hours` at simulation time
- Routing: SMS3 + LAG3 (storage-lag-weighted)

## API Reference

::: hydromodel.models.xaj_slw
