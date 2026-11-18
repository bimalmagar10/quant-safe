# Public-data check: steps-to-success vs binary success (VLAQuantBench LIBERO records)

- Quantized cells with >=30 both-success pairs: **267**
- of these, cells with no detectable success change (McNemar p > 0.05, uncorrected): **181**
- of those success-flat cells, cells where steps-to-success shifts (Wilcoxon, Holm-corrected p < 0.05 across all cells): **55** (30%), of which 43 are slower and 12 faster
- median |log steps ratio| in success-flat cells: 0.0097
- full-precision replicate cells found (noise floor): **4**

## Noise floor: baseline-vs-baseline replicates

| suite          | model   | cell                    |   n_both_success |   steps_ratio |   steps_ratio_lo |   steps_ratio_hi |   p_steps |
|:---------------|:--------|:------------------------|-----------------:|--------------:|-----------------:|-----------------:|----------:|
| libero_object  | pi05    | seed1-rtn-BASELINE-e2e  |              195 |        0.9908 |           0.9706 |           1.0077 |    0.5159 |
| libero_object  | pi05    | seed2-rtn-BASELINE-e2e  |              198 |        0.9812 |           0.9568 |           1.0007 |    0.0332 |
| libero_spatial | pi05    | steps1-rtn-BASELINE-e2e |              196 |        1.0013 |           0.9926 |           1.0107 |    0.9494 |
| libero_spatial | pi05    | steps2-rtn-BASELINE-e2e |              193 |        1.0113 |           0.9969 |           1.0322 |    0.1678 |

## Per model: success-flat cells and how many show a steps shift

| model       |   flat_cells |   steps_shift_cells |   median_steps_ratio |   max_steps_ratio |
|:------------|-------------:|--------------------:|---------------------:|------------------:|
| openvla_oft |       39.000 |              13.000 |                1.001 |             1.057 |
| pi0         |       31.000 |               3.000 |                0.990 |             1.053 |
| pi05        |       74.000 |              35.000 |                1.009 |             1.177 |
| xvla        |       37.000 |               4.000 |                1.001 |             1.052 |

## Success-flat cells with the largest steps shift

| suite          | model       | cell                                  |   sr_fp |   sr_q |   p_success |   n_both_success |   steps_ratio |   steps_ratio_lo |   steps_ratio_hi |   steps_dz |   p_steps_holm |
|:---------------|:------------|:--------------------------------------|--------:|-------:|------------:|-----------------:|--------------:|-----------------:|-----------------:|-----------:|---------------:|
| libero_spatial | pi05        | rtn-W3G128SYM-e2e                     |   0.99  |  0.96  |       0.109 |              190 |         1.18  |            1.16  |            1.2   |      2.75  |       7.87e-30 |
| libero_spatial | pi05        | rtn-W4A6-e2e                          |   0.99  |  0.975 |       0.375 |              194 |         1.09  |            1.07  |            1.1   |      1.27  |       3.33e-28 |
| libero_spatial | pi05        | rtn-W2-ve                             |   0.99  |  0.995 |       1     |              197 |         1.08  |            1.06  |            1.11  |      1.18  |       4.99e-27 |
| libero_spatial | pi05        | rtn-W3G64SYM-e2e                      |   0.99  |  0.975 |       0.453 |              193 |         1.13  |            1.1   |            1.15  |      1.15  |       2.08e-28 |
| libero_spatial | pi05        | rtn-W3-e2e                            |   0.99  |  0.99  |       1     |              196 |         1.05  |            1.04  |            1.07  |      0.796 |       1.21e-23 |
| libero_spatial | pi05        | rtn-W3G64ASYM-e2e                     |   0.99  |  0.97  |       0.289 |              192 |         1.04  |            1.03  |            1.06  |      0.73  |       8.59e-21 |
| libero_spatial | pi05        | rtn-W4A6-ah                           |   0.99  |  0.97  |       0.125 |              194 |         1.05  |            1.03  |            1.07  |      0.717 |       3.41e-22 |
| libero_spatial | pi05        | ablate-W4A4-ah-oproj_adarms           |   0.99  |  0.965 |       0.125 |              192 |         1.07  |            1.05  |            1.1   |      0.628 |       1.07e-24 |
| libero_goal    | pi05        | rtn-W3-e2e                            |   0.94  |  0.955 |       0.648 |              180 |         1.07  |            1.04  |            1.09  |      0.586 |       7.65e-16 |
| libero_spatial | pi05        | rtn-W4G0SYM-e2e                       |   0.99  |  0.98  |       0.688 |              194 |         1.02  |            1.01  |            1.04  |      0.492 |       7.41e-15 |
| libero_object  | pi05        | seed2-ablate-W4A4-ah-onlyadarms       |   0.99  |  0.97  |       0.219 |              193 |         1.07  |            1.04  |            1.12  |      0.464 |       6.46e-14 |
| libero_spatial | pi05        | steps1-ablate-W4A4-ah-only_action_out |   0.99  |  0.98  |       0.688 |              194 |         1.03  |            1.02  |            1.04  |      0.446 |       2.01e-12 |
| libero_spatial | pi05        | calib_all-W4A6-e2e                    |   0.99  |  0.98  |       0.625 |              195 |         1.03  |            1.02  |            1.05  |      0.419 |       1.52e-13 |
| libero_object  | pi05        | seed1-ablate-W4A4-ah-onlyadarms       |   0.99  |  0.96  |       0.109 |              190 |         1.06  |            1.03  |            1.09  |      0.419 |       1.8e-12  |
| libero_10      | openvla_oft | seed2-ablate-W3-e2e-no_ah_fc2         |   0.925 |  0.91  |       0.701 |              170 |         1.06  |            1.02  |            1.09  |      0.411 |       1.51e-10 |
| libero_spatial | pi05        | rtn-W4A6-llm                          |   0.99  |  0.975 |       0.453 |              193 |         1.04  |            1.02  |            1.07  |      0.406 |       1.37e-17 |
| libero_10      | openvla_oft | ablate-W3-e2e-no_any_fc2              |   0.925 |  0.93  |       1     |              173 |         1.05  |            1.01  |            1.08  |      0.391 |       7.41e-11 |
| libero_10      | openvla_oft | seed1-ablate-W3-e2e-no_ah_fc2         |   0.925 |  0.915 |       0.845 |              171 |         1.06  |            1.02  |            1.09  |      0.364 |       2.37e-09 |
| libero_10      | openvla_oft | ablate-W3-e2e-no_ah_fc2               |   0.925 |  0.935 |       0.839 |              174 |         1.05  |            1.01  |            1.08  |      0.363 |       1.56e-09 |
| libero_spatial | pi05        | rtn-W4G0ASYM-e2e                      |   0.99  |  0.975 |       0.375 |              194 |         1.04  |            1.02  |            1.07  |      0.348 |       1.02e-13 |
| libero_goal    | pi05        | rtn-W4A8-e2e                          |   0.94  |  0.965 |       0.302 |              183 |         1.03  |            1.01  |            1.06  |      0.316 |       7.62e-09 |
| libero_spatial | pi05        | rtn-W3-ve                             |   0.99  |  0.995 |       1     |              197 |         1.02  |            1.01  |            1.04  |      0.307 |       1.18e-09 |
| libero_spatial | pi05        | rtn-W4G128SYM-e2e                     |   0.99  |  0.98  |       0.688 |              194 |         1.01  |            1     |            1.03  |      0.305 |       2.32e-07 |
| libero_spatial | pi05        | rtn-W3-llm                            |   0.99  |  0.98  |       0.688 |              194 |         1.03  |            1.01  |            1.06  |      0.293 |       3.56e-13 |
| libero_spatial | pi05        | calib_ah-W4A4-ah-attn_adarms          |   0.99  |  0.975 |       0.453 |              193 |         0.966 |            0.944 |            0.997 |     -0.274 |       7.09e-18 |

## End-to-end RTN presets (the headline cells)

| model       | suite          | preset    |   sr_fp |   sr_q |   p_success |   n_both_success |   steps_ratio |   steps_ratio_lo |   steps_ratio_hi |   p_steps_holm |
|:------------|:---------------|:----------|--------:|-------:|------------:|-----------------:|--------------:|-----------------:|-----------------:|---------------:|
| openvla_oft | libero_10      | W3        |   0.925 |  0.08  |    2.67e-51 |               16 |         1.31  |            1.22  |            1.44  |     nan        |
| openvla_oft | libero_10      | W4        |   0.925 |  0.885 |    0.2      |              166 |         1.03  |            0.99  |            1.06  |       0.00154  |
| openvla_oft | libero_10      | W4A4      |   0.925 |  0     |    4.08e-56 |                0 |       nan     |          nan     |          nan     |     nan        |
| openvla_oft | libero_10      | W4A8      |   0.925 |  0.76  |    1.96e-06 |              144 |         1.22  |            1.18  |            1.27  |       1.28e-21 |
| openvla_oft | libero_10      | W8        |   0.925 |  0.935 |    0.791    |              179 |         0.989 |            0.976 |            1     |       0.971    |
| openvla_oft | libero_10      | W8A8      |   0.925 |  0.64  |    3.23e-12 |              120 |         1.17  |            1.13  |            1.23  |       1.69e-13 |
| openvla_oft | libero_goal    | W3        |   0.96  |  0.715 |    1.91e-11 |              138 |         1.02  |            0.986 |            1.07  |       1        |
| openvla_oft | libero_goal    | W4        |   0.96  |  0.82  |    1.94e-06 |              160 |         1.04  |            0.973 |            1.11  |       1        |
| openvla_oft | libero_goal    | W4A4      |   0.96  |  0     |    3.19e-58 |                0 |       nan     |          nan     |          nan     |     nan        |
| openvla_oft | libero_goal    | W4A8      |   0.96  |  0.02  |    5.1e-57  |                4 |       nan     |          nan     |          nan     |     nan        |
| openvla_oft | libero_goal    | W8        |   0.96  |  0.975 |    0.25     |              192 |         0.998 |            0.979 |            1.02  |       1        |
| openvla_oft | libero_goal    | W8A8      |   0.96  |  0.13  |    2.14e-50 |               26 |         1.29  |            0.967 |            2.14  |     nan        |
| openvla_oft | libero_object  | W3        |   0.975 |  0.745 |    2.27e-12 |              147 |         1.1   |            1.02  |            1.21  |       0.000194 |
| openvla_oft | libero_object  | W4        |   0.975 |  0.935 |    0.0386   |              185 |         1.02  |            0.983 |            1.06  |       1        |
| openvla_oft | libero_object  | W4A4      |   0.975 |  0     |    3.98e-59 |                0 |       nan     |          nan     |          nan     |     nan        |
| openvla_oft | libero_object  | W4A8      |   0.975 |  0.025 |    6.15e-56 |                4 |       nan     |          nan     |          nan     |     nan        |
| openvla_oft | libero_object  | W8        |   0.975 |  0.98  |    1        |              194 |         1.02  |            1     |            1.04  |       0.0115   |
| openvla_oft | libero_object  | W8A8      |   0.975 |  0.025 |    1.27e-57 |                5 |         1.08  |            0.95  |            1.52  |     nan        |
| openvla_oft | libero_spatial | W3        |   1     |  0.78  |    1.14e-13 |              156 |         1.06  |            1.03  |            1.09  |       4.91e-11 |
| openvla_oft | libero_spatial | W4        |   1     |  0.96  |    0.00781  |              192 |         1.09  |            1.08  |            1.11  |       3.08e-30 |
| openvla_oft | libero_spatial | W4A4      |   1     |  0     |    1.24e-60 |                0 |       nan     |          nan     |          nan     |     nan        |
| openvla_oft | libero_spatial | W4A6      |   1     |  0.235 |    1.75e-46 |               47 |         1.3   |            1.21  |            1.4   |       3.01e-12 |
| openvla_oft | libero_spatial | W4A8      |   1     |  0.215 |    1.09e-47 |               43 |         1.4   |            1.3   |            1.51  |       4.73e-11 |
| openvla_oft | libero_spatial | W8        |   1     |  0.995 |    1        |              199 |         0.999 |            0.995 |            1     |       1        |
| openvla_oft | libero_spatial | W8A8      |   1     |  0.36  |    5.88e-39 |               72 |         1.28  |            1.18  |            1.44  |       7.04e-11 |
| pi0         | libero_10      | W3        |   0.485 |  0.4   |    0.043    |               57 |         1.05  |            0.955 |            1.14  |       1        |
| pi0         | libero_10      | W4        |   0.485 |  0.41  |    0.0912   |               55 |         1.01  |            0.947 |            1.08  |       1        |
| pi0         | libero_10      | W4A4      |   0.485 |  0.31  |    0.000224 |               36 |         1.1   |            0.985 |            1.21  |       1        |
| pi0         | libero_10      | W4A8      |   0.485 |  0.41  |    0.0722   |               59 |         1.01  |            0.936 |            1.08  |       1        |
| pi0         | libero_10      | W8        |   0.485 |  0.475 |    0.883    |               73 |         0.998 |            0.963 |            1.03  |       1        |
| pi0         | libero_10      | W8A8      |   0.485 |  0.45  |    0.337    |               74 |         0.991 |            0.965 |            1.02  |       1        |
| pi0         | libero_goal    | W3        |   0.845 |  0.645 |    2.78e-07 |              118 |         1.12  |            1.06  |            1.22  |       3.81e-11 |
| pi0         | libero_goal    | W4        |   0.845 |  0.79  |    0.0708   |              148 |         1     |            0.98  |            1.03  |       1        |
| pi0         | libero_goal    | W4A4      |   0.845 |  0.58  |    5.92e-10 |              104 |         1.01  |            0.973 |            1.05  |       1        |
| pi0         | libero_goal    | W4A8      |   0.845 |  0.79  |    0.0801   |              147 |         1     |            0.982 |            1.04  |       1        |
| pi0         | libero_goal    | W8        |   0.845 |  0.83  |    0.549    |              162 |         1     |            0.994 |            1.01  |       1        |
| pi0         | libero_goal    | W8A8      |   0.845 |  0.81  |    0.0923   |              159 |         1     |            0.994 |            1.01  |       1        |
| pi0         | libero_object  | W3        |   0.875 |  0.76  |    0.000606 |              142 |         1.01  |            0.971 |            1.05  |       1        |
| pi0         | libero_object  | W4        |   0.875 |  0.905 |    0.307    |              166 |         1.02  |            0.996 |            1.05  |       0.0849   |
| pi0         | libero_object  | W4A4      |   0.875 |  0.74  |    0.000584 |              132 |         1.06  |            1.02  |            1.1   |       0.0136   |
| pi0         | libero_object  | W4A8      |   0.875 |  0.885 |    0.856    |              161 |         1.03  |            0.996 |            1.06  |       0.0324   |
| pi0         | libero_object  | W8        |   0.875 |  0.865 |    0.688    |              171 |         1     |            0.99  |            1.01  |       1        |
| pi0         | libero_object  | W8A8      |   0.875 |  0.885 |    0.791    |              169 |         1     |            0.985 |            1.02  |       1        |
| pi0         | libero_spatial | W3        |   0.78  |  0.71  |    0.103    |              117 |         1.05  |            0.991 |            1.11  |       3.31e-07 |
| pi0         | libero_spatial | W4        |   0.78  |  0.69  |    0.0198   |              120 |         1.01  |            0.957 |            1.08  |       0.29     |
| pi0         | libero_spatial | W4A4      |   0.78  |  0.61  |    4.45e-05 |              105 |         1.06  |            0.986 |            1.16  |       1        |
| pi0         | libero_spatial | W4A8      |   0.78  |  0.645 |    0.00046  |              114 |         1.01  |            0.959 |            1.05  |       0.0986   |
| pi0         | libero_spatial | W8        |   0.78  |  0.775 |    1        |              150 |         0.994 |            0.962 |            1.03  |       1        |
| pi0         | libero_spatial | W8A8      |   0.78  |  0.745 |    0.21     |              141 |         0.976 |            0.943 |            0.999 |       1        |
| pi05        | libero_10      | W3        |   0.95  |  0.945 |    1        |              180 |         1.03  |            1.01  |            1.06  |       7.68e-06 |
| pi05        | libero_10      | W4        |   0.95  |  0.94  |    0.774    |              183 |         1.01  |            0.993 |            1.03  |       1        |
| pi05        | libero_10      | W4A4      |   0.95  |  0     |    1.27e-57 |                0 |       nan     |          nan     |          nan     |     nan        |
| pi05        | libero_10      | W4A8      |   0.95  |  0.965 |    0.581    |              185 |         1.01  |            0.991 |            1.03  |       1        |
| pi05        | libero_10      | W8        |   0.95  |  0.965 |    0.508    |              187 |         1     |            0.987 |            1.03  |       1        |
| pi05        | libero_10      | W8A8      |   0.95  |  0.935 |    0.508    |              184 |         0.996 |            0.977 |            1.01  |       1        |
| pi05        | libero_goal    | W3        |   0.94  |  0.955 |    0.648    |              180 |         1.07  |            1.04  |            1.09  |       7.65e-16 |
| pi05        | libero_goal    | W4        |   0.94  |  0.965 |    0.302    |              183 |         1.03  |            1.01  |            1.06  |       3.82e-07 |
| pi05        | libero_goal    | W4A4      |   0.94  |  0     |    5.1e-57  |                0 |       nan     |          nan     |          nan     |     nan        |
| pi05        | libero_goal    | W4A8      |   0.94  |  0.965 |    0.302    |              183 |         1.03  |            1.01  |            1.06  |       7.62e-09 |
| pi05        | libero_goal    | W8        |   0.94  |  0.96  |    0.219    |              187 |         0.984 |            0.967 |            0.997 |       1        |
| pi05        | libero_goal    | W8A8      |   0.94  |  0.97  |    0.0703   |              187 |         0.998 |            0.98  |            1.02  |       1        |
| pi05        | libero_object  | W3        |   0.99  |  1     |    0.5      |              198 |         1.03  |            0.995 |            1.05  |       1.11e-08 |
| pi05        | libero_object  | W4        |   0.99  |  0.99  |    1        |              196 |         1.01  |            0.985 |            1.03  |       1        |
| pi05        | libero_object  | W4A4      |   0.99  |  0     |    4.98e-60 |                0 |       nan     |          nan     |          nan     |     nan        |
| pi05        | libero_object  | W4A8      |   0.99  |  1     |    0.5      |              198 |         1.01  |            0.99  |            1.04  |       0.17     |
| pi05        | libero_object  | W8        |   0.99  |  0.99  |    1        |              197 |         0.995 |            0.977 |            1.01  |       1        |
| pi05        | libero_object  | W8A8      |   0.99  |  0.99  |    1        |              197 |         1     |            0.985 |            1.02  |       1        |
| pi05        | libero_spatial | W2        |   0.99  |  0     |    4.98e-60 |                0 |       nan     |          nan     |          nan     |     nan        |
| pi05        | libero_spatial | W3        |   0.99  |  0.99  |    1        |              196 |         1.05  |            1.04  |            1.07  |       1.21e-23 |
| pi05        | libero_spatial | W3G0ASYM  |   0.99  |  0.2   |    5.47e-48 |               40 |         1.57  |            1.37  |            1.74  |       3.62e-10 |
| pi05        | libero_spatial | W3G0SYM   |   0.99  |  0     |    4.98e-60 |                0 |       nan     |          nan     |          nan     |     nan        |
| pi05        | libero_spatial | W3G128SYM |   0.99  |  0.96  |    0.109    |              190 |         1.18  |            1.16  |            1.2   |       7.87e-30 |
| pi05        | libero_spatial | W3G64ASYM |   0.99  |  0.97  |    0.289    |              192 |         1.04  |            1.03  |            1.06  |       8.59e-21 |
| pi05        | libero_spatial | W3G64SYM  |   0.99  |  0.975 |    0.453    |              193 |         1.13  |            1.1   |            1.15  |       2.08e-28 |
| pi05        | libero_spatial | W4        |   0.99  |  0.975 |    0.375    |              194 |         1.01  |            0.999 |            1.02  |       0.0674   |
| pi05        | libero_spatial | W4A4      |   0.99  |  0     |    4.98e-60 |                0 |       nan     |          nan     |          nan     |     nan        |
| pi05        | libero_spatial | W4A6      |   0.99  |  0.975 |    0.375    |              194 |         1.09  |            1.07  |            1.1   |       3.33e-28 |
| pi05        | libero_spatial | W4A8      |   0.99  |  0.99  |    1        |              196 |         1.02  |            1     |            1.04  |       0.000634 |
| pi05        | libero_spatial | W4G0ASYM  |   0.99  |  0.975 |    0.375    |              194 |         1.04  |            1.02  |            1.07  |       1.02e-13 |
| pi05        | libero_spatial | W4G0SYM   |   0.99  |  0.98  |    0.688    |              194 |         1.02  |            1.01  |            1.04  |       7.41e-15 |
| pi05        | libero_spatial | W4G128SYM |   0.99  |  0.98  |    0.688    |              194 |         1.01  |            1     |            1.03  |       2.32e-07 |
| pi05        | libero_spatial | W4G64ASYM |   0.99  |  0.975 |    0.453    |              193 |         1.01  |            1     |            1.02  |       0.0451   |
| pi05        | libero_spatial | W4G64SYM  |   0.99  |  0.97  |    0.289    |              192 |         1.01  |            1     |            1.03  |       0.00472  |
| pi05        | libero_spatial | W8        |   0.99  |  0.985 |    1        |              196 |         1     |            0.995 |            1.01  |       1        |
| pi05        | libero_spatial | W8A8      |   0.99  |  0.985 |    1        |              195 |         1     |            0.997 |            1.01  |       1        |
| xvla        | libero_10      | W3        |   0.979 |  0.947 |    0.146    |              177 |         1     |            0.987 |            1.02  |       0.645    |
| xvla        | libero_10      | W4        |   0.979 |  0.958 |    0.289    |              180 |         1.01  |            0.994 |            1.03  |       1        |
| xvla        | libero_10      | W4A4      |   0.979 |  0     |    2.04e-56 |                0 |       nan     |          nan     |          nan     |     nan        |
| xvla        | libero_10      | W4A8      |   0.979 |  0.968 |    0.625    |              183 |         0.996 |            0.963 |            1.03  |       1        |
| xvla        | libero_10      | W8        |   0.979 |  0.974 |    1        |              184 |         1     |            0.985 |            1.02  |       1        |
| xvla        | libero_10      | W8A8      |   0.979 |  0.984 |    1        |              184 |         0.994 |            0.967 |            1.02  |       1        |
| xvla        | libero_goal    | W3        |   0.974 |  0.974 |    1        |              181 |         0.988 |            0.954 |            1.02  |       0.0545   |
| xvla        | libero_goal    | W4        |   0.974 |  0.968 |    1        |              180 |         1     |            0.966 |            1.04  |       1        |
| xvla        | libero_goal    | W4A4      |   0.974 |  0     |    4.08e-56 |                0 |       nan     |          nan     |          nan     |     nan        |
| xvla        | libero_goal    | W4A8      |   0.974 |  0.979 |    1        |              183 |         0.986 |            0.952 |            1.01  |       1        |
| xvla        | libero_goal    | W8        |   0.974 |  0.963 |    0.625    |              182 |         0.983 |            0.95  |            1     |       1        |
| xvla        | libero_goal    | W8A8      |   0.974 |  0.963 |    0.727    |              180 |         0.988 |            0.956 |            1.01  |       1        |
| xvla        | libero_object  | W3        |   0.979 |  0.968 |    0.727    |              181 |         1.02  |            1     |            1.05  |       0.714    |
| xvla        | libero_object  | W4        |   0.979 |  0.979 |    1        |              184 |         0.994 |            0.989 |            0.999 |       0.0104   |
| xvla        | libero_object  | W4A4      |   0.979 |  0     |    2.04e-56 |                0 |       nan     |          nan     |          nan     |     nan        |
| xvla        | libero_object  | W4A8      |   0.979 |  0.984 |    1        |              184 |         0.993 |            0.982 |            1     |       0.281    |
| xvla        | libero_object  | W8        |   0.979 |  0.979 |    1        |              184 |         0.999 |            0.995 |            1     |       1        |
| xvla        | libero_object  | W8A8      |   0.979 |  0.979 |    1        |              184 |         0.997 |            0.978 |            1.02  |       1        |
| xvla        | libero_spatial | W3        |   0.968 |  0.968 |    1        |              181 |         0.994 |            0.987 |            1     |       0.0017   |
| xvla        | libero_spatial | W4        |   0.968 |  0.958 |    0.688    |              180 |         1     |            0.995 |            1.03  |       1        |
| xvla        | libero_spatial | W4A4      |   0.968 |  0     |    8.16e-56 |                0 |       nan     |          nan     |          nan     |     nan        |
| xvla        | libero_spatial | W4A8      |   0.968 |  0.963 |    1        |              181 |         1.01  |            0.997 |            1.05  |       1        |
| xvla        | libero_spatial | W8        |   0.968 |  0.974 |    1        |              184 |         1     |            0.999 |            1     |       1        |
| xvla        | libero_spatial | W8A8      |   0.968 |  0.963 |    1        |              182 |         1.01  |            0.996 |            1.05  |       1        |
