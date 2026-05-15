//! Deterministic synthetic fixtures for `crucible-models`.
//!
//! These generators are used in unit tests, training CLIs, and
//! showcase scenarios. They produce learnable classification tasks
//! with realistic shape — they do not model any real-world domain
//! and should never be used to make actual decisions. Each generator
//! is fully deterministic given its seed.

use ndarray::{Array1, Array2};
use rand::{Rng as _, SeedableRng as _, rngs::StdRng};

/// Synthetic loan-default classification dataset.
///
/// Features (columns):
///
/// | idx | name                | range / domain        |
/// |----:|---------------------|-----------------------|
/// |  0  | `credit_score`      | `[300, 850]`          |
/// |  1  | `annual_income`     | `[10_000, 300_000]`   |
/// |  2  | `debt_to_income`    | `[0.0, 1.0]`          |
/// |  3  | `loan_amount`       | `[1_000, 500_000]`    |
/// |  4  | `employment_years`  | `[0, 40]`             |
/// |  5  | `has_prior_default` | `{0.0, 1.0}`          |
///
/// Label is `defaulted ∈ {0, 1}`, derived from a logistic function
/// over the features with small per-sample noise. Default rate is
/// driven by low credit score, high debt-to-income, prior defaults,
/// and a high loan-to-income ratio.
pub fn loan_default(n_samples: usize, seed: u64) -> (Array2<f64>, Array1<usize>) {
    const N_FEATURES: usize = 6;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut features = Vec::with_capacity(n_samples * N_FEATURES);
    let mut labels = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let credit_score: f64 = rng.gen_range(300.0..=850.0);
        let annual_income: f64 = rng.gen_range(10_000.0..=300_000.0);
        let debt_to_income: f64 = rng.gen_range(0.0..=1.0);
        let loan_amount: f64 = rng.gen_range(1_000.0..=500_000.0);
        let employment_years: f64 = rng.gen_range(0.0..=40.0);
        let has_prior_default: f64 = if rng.gen_bool(0.15) { 1.0 } else { 0.0 };

        // Logistic over features. Higher credit score, more employment
        // years, and higher income reduce default probability; high
        // debt-to-income, prior default, and high loan-to-income ratio
        // increase it. Small per-sample noise on the logit.
        let loan_to_income = loan_amount / annual_income.max(1.0);
        let logit = -3.0
            + 0.012 * (700.0 - credit_score)
            + 4.0 * debt_to_income
            + 2.0 * has_prior_default
            + 0.6 * loan_to_income
            - 0.02 * employment_years
            + rng.gen_range(-0.4..0.4);
        let p_default = 1.0 / (1.0 + (-logit).exp());
        let defaulted = usize::from(rng.gen_range(0.0_f64..1.0_f64) < p_default);

        features.push(credit_score);
        features.push(annual_income);
        features.push(debt_to_income);
        features.push(loan_amount);
        features.push(employment_years);
        features.push(has_prior_default);
        labels.push(defaulted);
    }

    let features = Array2::from_shape_vec((n_samples, N_FEATURES), features)
        .expect("shape known correct from construction");
    let labels = Array1::from_vec(labels);
    (features, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loan_default_is_deterministic_under_fixed_seed() {
        let (a_x, a_y) = loan_default(50, 42);
        let (b_x, b_y) = loan_default(50, 42);
        assert_eq!(a_x, b_x);
        assert_eq!(a_y, b_y);
    }

    #[test]
    fn loan_default_shape_matches_request() {
        let (features, labels) = loan_default(100, 0);
        assert_eq!(features.shape(), &[100, 6]);
        assert_eq!(labels.len(), 100);
    }

    #[test]
    fn loan_default_has_both_classes_in_realistic_balance() {
        let (_, labels) = loan_default(500, 7);
        let n_def = labels.iter().filter(|&&y| y == 1).count();
        assert!(
            n_def > 50 && n_def < 450,
            "expected meaningful class balance, got {n_def}/500 default-positive"
        );
    }

    #[test]
    fn loan_default_features_within_documented_ranges() {
        let (features, _) = loan_default(200, 1);
        for row in features.rows() {
            assert!((300.0..=850.0).contains(&row[0]));
            assert!((10_000.0..=300_000.0).contains(&row[1]));
            assert!((0.0..=1.0).contains(&row[2]));
            assert!((1_000.0..=500_000.0).contains(&row[3]));
            assert!((0.0..=40.0).contains(&row[4]));
            assert!(row[5] == 0.0 || row[5] == 1.0);
        }
    }
}
