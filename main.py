import pandas as pd
import numpy as np
import scipy.stats
from scipy import stats

# Load data
data = pd.read_csv('data.csv')

# Get prices in correct format
prices = data.iloc[:, 3].dropna().apply(lambda x: float(str(x).replace(';', '.')))
dates = data.iloc[:, 2].dropna()

# 1. Calculate simple returns
returns = prices.pct_change().dropna()


# 2. Filter out zero returns
filtered_returns = returns[returns != 0]


n_test_result = scipy.stats.normaltest(filtered_returns)
print("Normality test result", n_test_result)

# Basic stats
n = len(filtered_returns)
sample_mean = np.mean(filtered_returns)
sample_var = np.var(filtered_returns, ddof=1)
sample_std = np.std(filtered_returns, ddof=1)
se = sample_std / np.sqrt(n)


# 3. Hypothesis test: mean = 0
# t_stat_mean = (sample_mean - 0) / (sample_std / np.sqrt(n))
# p_value_mean = 2 * (1 - stats.t.cdf(abs(t_stat_mean), df=n-1))
t_stat_mean, p_value_mean = stats.ttest_1samp(filtered_returns, 0)
mean_0_hyp_rejected = True if p_value_mean < 0.05 else False
print(f"Mean test t-statistic: {t_stat_mean:.9f}, p-value: {p_value_mean:.9f}, rejected: {mean_0_hyp_rejected}")


# 4.1. Hypothesis test: variance = 1, with unknown mean
sigma0_squared = 1  # H0: variance = 1
squared_diffs_unknown_mean = np.sum((filtered_returns - sample_mean) ** 2)
chi2_stat_sample_mean = squared_diffs_unknown_mean / sigma0_squared
p_value_sample_mean = 2 * min(stats.chi2.cdf(chi2_stat_sample_mean, df=n), 1 - stats.chi2.cdf(chi2_stat_sample_mean, df=n))
variance_1_unknown_mean_hyp_rejected = True if p_value_sample_mean < 0.05 else False
print(f"Variance test (sample mean={sample_mean:.9f}) chi2-statistic: {chi2_stat_sample_mean:.9f}, "
      f"p-value: {p_value_sample_mean:.9f}, rejected: {variance_1_unknown_mean_hyp_rejected}")

# 4.2. Hypothesis test: variance = 1, with mean = 0
known_mean = 0
squared_diffs_known_mean = np.sum((filtered_returns - known_mean) ** 2)
chi2_stat_known_mean = squared_diffs_known_mean / sigma0_squared
p_value_known_mean = 2 * min(stats.chi2.cdf(chi2_stat_known_mean, df=n), 1 - stats.chi2.cdf(chi2_stat_known_mean, df=n))
variance_1_known_mean_hyp_rejected = True if p_value_known_mean < 0.05 else False
print(f"Variance test (mean known=0) chi2-statistic: {chi2_stat_known_mean:.4f}, "
      f"p-value: {p_value_known_mean:.9f}, rejected: {variance_1_known_mean_hyp_rejected}")


# 5. Confidence interval for the mean
z_crit = stats.norm.ppf(0.975) # twosided
ci_mean = (sample_mean - z_crit * sample_std, sample_mean + z_crit * sample_std)
print(f"95% confidence interval for mean: ({ci_mean[0]:.9f}, {ci_mean[1]:.9f})")


# 6.1. Confidence interval for the variance, unknown mean
se_var = np.sqrt((2 * sample_var ** 2) / (n - 1))
z_crit = stats.norm.ppf(0.975)  # 95% confidence
ci_var_norm = (sample_var - z_crit * se_var, sample_var + z_crit * se_var)
print(f"95% CI for variance (normal approx): ({ci_var_norm[0]:.9f}, {ci_var_norm[1]:.9f})")

# 6.2. Confidence interval for variance, known mean
alpha = 0.95
chi2_lower_known = stats.chi2.ppf(alpha/2, df=n)
chi2_upper_known = stats.chi2.ppf(1 - alpha/2, df=n)
ci_var_known_mean = (squared_diffs_known_mean / chi2_upper_known, squared_diffs_known_mean / chi2_lower_known)
print(f"95% CI for variance (mean known=0): ({ci_var_known_mean[0]:.9f}, {ci_var_known_mean[1]:.9f})")




# V1: visualize price sequence
def plot_sequence():
    import datetime as dt
    import matplotlib.pyplot as plt

    x = [dt.datetime.strptime(d,'%d.%m.%Y').date() for d in dates.values]
    y = prices

    plt.figure(figsize=(12, 6))
    plt.title('Baltic Benchmark Index (2005-05-01 - 2025-05-01)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Index')
    plt.plot(x,y)
    plt.gcf().autofmt_xdate()
    plt.grid(True)
    plt.savefig('baltic_benchmark_index_20_years.png', dpi=300, bbox_inches='tight')
    plt.show()


# V1_1: visualize returns
def plot_returns():
    import datetime as dt
    import matplotlib.pyplot as plt

    x = [dt.datetime.strptime(d,'%d.%m.%Y').date() for d in dates.values[:-1]]
    y = returns

    plt.figure(figsize=(12, 6))
    plt.title('Baltic Benchmark Index Returns (2005-05-01 - 2025-05-01)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.plot(x,y)
    plt.gcf().autofmt_xdate()
    plt.grid(True)
    plt.savefig('baltic_benchmark_index_returns_20_years.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_mean():
    import matplotlib.pyplot as plt
    # ------------------------------
    # 1. Mean Hypothesis Test
    # ------------------------------
    plt.hist(filtered_returns, bins=200, color='skyblue', label='Returns Histogram')

    # Mean and CI
    plt.axvline(sample_mean, color='red', linestyle='--', label=f'Sample mean: {sample_mean:.5f}')
    plt.axvspan(ci_mean[0], ci_mean[1], color='yellow', alpha=0.1, label='Mean CI')
    plt.axvline(ci_mean[0], color='darkorange', linestyle=':')
    plt.axvline(ci_mean[1], color='darkorange', linestyle=':')

    plt.title('Mean Test: Returns Distribution with 95% CI', fontsize=14)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.annotate(f'T-stat: {t_stat_mean:.5f}, \nMean p-value: {p_value_mean:.5f}\nRejected H0 (mean=0): {mean_0_hyp_rejected}',
                     xy=(0.7, 0.8), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle="round", fc="w"))
    plt.annotate(f'Confidence interval: \n({ci_mean[0]:.5f}, {ci_mean[1]:.5f})', xy=(0.7, 0.65), xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round", fc="w"))
    plt.tight_layout()
    plt.savefig('mean_test.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_variance_unknown_mean():
    import matplotlib.pyplot as plt
    plt.hist(filtered_returns, bins=200, color='skyblue', label='Returns Histogram')

    plt.axvspan(-np.sqrt(ci_var_norm[0]), np.sqrt(ci_var_norm[1]), color='yellow', alpha=0.1, label='Variance CI (±sqrt)')
    plt.axvline(-np.sqrt(ci_var_norm[0]), color='darkorange', linestyle=':')
    plt.axvline(np.sqrt(ci_var_norm[1]), color='darkorange', linestyle=':')

    plt.title('Variance Test with unknown mean', fontsize=14)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.annotate(f'Sample standard \ndeviation: {sample_std:.5f}',
                     xy=(0.1, 0.7), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle="round", fc="w"))
    plt.annotate(f'Chi2-stat: {chi2_stat_sample_mean:.5f}, \nVariance p: {p_value_sample_mean:.5f}\nRejected H0 (mean=0): {variance_1_known_mean_hyp_rejected}',
                     xy=(0.6, 0.8), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle="round", fc="w"))
    plt.tight_layout()
    plt.savefig('variance_test_unknown_mean.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_variance_known_mean():
    import matplotlib.pyplot as plt
    plt.hist(filtered_returns, bins=200, color='skyblue', label='Returns Histogram')

    plt.axvspan(-np.sqrt(ci_var_known_mean[0]), np.sqrt(ci_var_known_mean[1]), color='yellow', alpha=0.1, label='Variance CI (±sqrt)')
    plt.axvline(-np.sqrt(ci_var_known_mean[0]), color='darkorange', linestyle=':')
    plt.axvline(np.sqrt(ci_var_known_mean[1]), color='darkorange', linestyle=':')

    plt.title('Variance Test with known mean', fontsize=14)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.annotate(f'Sample standard \ndeviation: {sample_std:.5f}',
                     xy=(0.1, 0.7), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle="round", fc="w"))
    plt.annotate(f'Chi2-stat: {chi2_stat_known_mean:.5f}, \nVariance p: {p_value_known_mean:.5f}\nRejected H0 (mean=0): {variance_1_unknown_mean_hyp_rejected}',
                     xy=(0.6, 0.8), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle="round", fc="w"))
    plt.tight_layout()
    plt.savefig('variance_test_known_mean.png', dpi=300, bbox_inches='tight')
    plt.show()


plot_sequence()
plot_returns()
plot_mean()
plot_variance_unknown_mean()
plot_variance_known_mean()