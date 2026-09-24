# Experiment results

Regenerate with `make experiments` (all settings: `scripts/utils/config.py`).
n = 100, sigma = 0.1, 80/20 split, degrees 0-20, features standardized on the
training rows; repeated experiments use seeds 0-29 + 42. "True error" = E_x[(f_hat - f)^2]
+ sigma^2 on a dense grid (needs the known f).

| file | content | key numbers |
|---|---|---|
| e1_ols_degree.json | OLS degree selected by the bootstrap, 5/10-fold CV, a single split, and the true error | median degree 8 / 10 / 12 / 13 vs. true 14; true error at the selected degree 0.0179±0.0008 (bootstrap), 0.024±0.006 (5-fold) vs. 0.0131±0.0003 at best |
| e2_model_selection.json | 5-fold CV selection of OLS / Ridge / Lasso (own subgradient descent, Adam, 5000 it.), refit, test and true error, 31 data sets | medians: CV 0.0139/0.0129/0.0150, test 0.0134/0.0136/0.0134, true 0.0146/0.0137/0.0156; better than OLS (CV/test/true): Ridge 25/15/19, Lasso 11/12/10; Lasso median degree 8, never an exact zero |
| e3_lasso.json | subgradient descent with each optimizer vs. sklearn at the seed-42 CV choice (p = 6, lambda = 10^-2.5) and at p = 15, lambda = 1e-4 | p=6: gap levels off at 1e-6 (plain, AdaGrad) to 2e-3 (Adam); p=15: best gap after 5e4 it. 1.2e-2 (Adam), coefficients off by up to 0.5; sklearn has 1 and 5 exact zeros, ours none |
| e4_optimizers.json | iterations to reach 1e-3 relative parameter error, tuned learning rate | p=5 OLS: plain 3166, momentum 126, AdaGrad 3354, RMSprop 3181, Adam 337; OLS p>=10: none within 1e5 |
| e5_autodiff.json | analytical vs. JAX gradients | max relative difference 5e-16; JAX d|x|/dx at 0 = 1 |
