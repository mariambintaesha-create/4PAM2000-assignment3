"""
Assignment 3: Clustering and Fitting with World Bank Climate Data
4PAM2000 Data Science Lab

Author : [Your Name]
SNR    : [Your Student Number]
GitHub : [Your GitHub Repository URL]

This script downloads World Bank climate-related indicators, applies
K-Means clustering to group countries by energy and emissions profiles,
and fits an exponential growth model to CO2 emission time series,
producing predictions with confidence intervals.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

import errors  # provided err_ranges / error_prop helper module


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# World Bank indicator codes
IND_CO2_PC = "EN.ATM.CO2E.PC"          # CO2 emissions (metric tons per capita)
IND_RENEW = "EG.FEC.RNEW.ZS"      # Renewable energy (% of total consumption)
IND_GDP_PC = "NY.GDP.PCAP.CD"     # GDP per capita (current US$)
IND_ACCESS = "EG.ELC.ACCS.ZS"     # Access to electricity (% of population)

CLUSTER_YEAR = 2020    # snapshot year used for clustering
FIT_COUNTRY = "CHN"    # country used for the fitting analysis
PRED_YEAR = 2045       # target prediction year
N_CLUSTERS = 4         # number of K-Means clusters
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data download helpers
# ---------------------------------------------------------------------------

def download_wb_indicator(indicator, start=1990, end=2022):
    """
    Download a single World Bank indicator via the public JSON API.

    Parameters
    ----------
    indicator : str
        World Bank indicator code.
    start, end : int
        Year range (inclusive).

    Returns
    -------
    pd.DataFrame
        Columns: country_code, country_name, year, value.
    """
    base = "https://api.worldbank.org/v2/country/all/indicator"
    url = (
        f"{base}/{indicator}"
        f"?date={start}:{end}&format=json&per_page=20000"
    )
    raw = pd.read_json(url)

    # World Bank wraps data in a two-element list; element [1] is the payload
    records = raw.iloc[1, 0]
    rows = []
    for rec in records:
        rows.append({
            "country_code": rec["countryiso3code"],
            "country_name": rec["country"]["value"],
            "year": int(rec["date"]),
            "value": rec["value"],
        })

    df = pd.DataFrame(rows)
    df = df[df["value"].notna()]
    return df


def build_cluster_snapshot(year=CLUSTER_YEAR):
    """
    Build a country-level snapshot dataframe for clustering.

    Downloads CO2 per capita, renewable energy share, GDP per capita,
    and electricity access for the chosen year, merges them, and drops
    countries with missing values.

    Parameters
    ----------
    year : int
        The snapshot year.

    Returns
    -------
    pd.DataFrame
        One row per country, columns: country_code, country_name,
        co2_pc, renew_pct, gdp_pc, elec_access.
    """
    print(f"Downloading World Bank data for snapshot year {year} ...")

    def _get(ind):
        df = download_wb_indicator(ind, start=year, end=year)
        df = df[["country_code", "country_name", "value"]].copy()
        df = df[df["country_code"].str.len() == 3]   # drop aggregates
        return df

    co2 = _get(IND_CO2_PC).rename(columns={"value": "co2_pc"})
    ren = _get(IND_RENEW).rename(columns={"value": "renew_pct"})
    gdp = _get(IND_GDP_PC).rename(columns={"value": "gdp_pc"})
    acc = _get(IND_ACCESS).rename(columns={"value": "elec_access"})

    merged = (
        co2
        .merge(
            ren[["country_code", "renew_pct"]],
            on="country_code", how="inner"
        )
        .merge(
            gdp[["country_code", "gdp_pc"]],
            on="country_code", how="inner"
        )
        .merge(
            acc[["country_code", "elec_access"]],
            on="country_code", how="inner"
        )
    )
    merged = merged.dropna().reset_index(drop=True)
    n = len(merged)
    print(f"  {n} countries retained after merging and dropping NaN.")
    return merged


def build_timeseries(country_code=FIT_COUNTRY, start=1990, end=2022):
    """
    Download annual CO2 per capita time series for a single country.

    Parameters
    ----------
    country_code : str
        ISO 3-letter country code.
    start, end : int
        Year range.

    Returns
    -------
    pd.DataFrame
        Columns: year, co2_pc, sorted ascending by year.
    """
    print(f"Downloading CO2 time series for {country_code} ...")
    df = download_wb_indicator(IND_CO2_PC, start=start, end=end)
    df = df[df["country_code"] == country_code].copy()
    df = df[["year", "value"]].rename(columns={"value": "co2_pc"})
    df = df.sort_values("year").reset_index(drop=True)
    df = df.dropna()
    return df


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_clustering(snapshot_df, n_clusters=N_CLUSTERS):
    """
    Normalise the four features and fit K-Means clustering.

    Parameters
    ----------
    snapshot_df : pd.DataFrame
        Output of build_cluster_snapshot().
    n_clusters : int
        Number of clusters for K-Means.

    Returns
    -------
    labelled_df : pd.DataFrame
        snapshot_df with an additional 'cluster' column (original values).
    centres_orig : np.ndarray
        Cluster centres back-transformed to original scale.
    scaler : StandardScaler
        Fitted scaler (needed for back-transform).
    features : list of str
        Feature column names used for clustering.
    """
    features = ["co2_pc", "renew_pct", "gdp_pc", "elec_access"]

    # Work on a copy so the original dataframe is not modified
    cluster_df = snapshot_df[features].copy()

    scaler = StandardScaler()
    normalised = scaler.fit_transform(cluster_df)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(normalised)

    # Back-transform cluster centres to original scale for display
    centres_orig = scaler.inverse_transform(km.cluster_centers_)

    labelled_df = snapshot_df.copy()
    labelled_df["cluster"] = labels
    return labelled_df, centres_orig, scaler, features


def plot_clusters(labelled_df, centres_orig, year=CLUSTER_YEAR):
    """
    Produce a 2-D scatter plot of CO2 per capita vs GDP per capita,
    coloured by cluster, with cluster centres marked.

    Parameters
    ----------
    labelled_df : pd.DataFrame
        Output of run_clustering() — must contain co2_pc, gdp_pc, cluster.
    centres_orig : np.ndarray
        Back-transformed cluster centres, shape (n_clusters, n_features).
        Column order: co2_pc, renew_pct, gdp_pc, elec_access.
    year : int
        Snapshot year shown in title.
    """
    colours = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    n_clusters = len(centres_orig)
    cluster_labels = [f"Cluster {i}" for i in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(10, 7))

    for cl in range(n_clusters):
        mask = labelled_df["cluster"] == cl
        subset = labelled_df[mask]
        ax.scatter(
            subset["gdp_pc"] / 1000,
            subset["co2_pc"],
            c=colours[cl],
            alpha=0.65,
            s=50,
            label=cluster_labels[cl],
            zorder=2,
        )

    # Cluster centres (back-transformed): col 0 = co2_pc, col 2 = gdp_pc
    for i, centre in enumerate(centres_orig):
        ax.scatter(
            centre[2] / 1000,
            centre[0],
            c=colours[i],
            s=220,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

    ax.set_xlabel("GDP per capita (thousand USD)", fontsize=13)
    ax.set_ylabel("CO2 emissions per capita (metric tons)", fontsize=13)
    ax.set_title(
        f"Country clusters by energy and emissions profile ({year})\n"
        "Stars mark cluster centres",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig("clustering_result.png", dpi=150)
    plt.show()
    print("  Saved: clustering_result.png")


def print_cluster_summary(labelled_df, centres_orig, features):
    """
    Print a readable summary table of cluster centres and member counts.

    Parameters
    ----------
    labelled_df : pd.DataFrame
        Labelled data from run_clustering().
    centres_orig : np.ndarray
        Back-transformed cluster centres.
    features : list of str
        Feature names matching columns in centres_orig.
    """
    print("\n--- Cluster summary (back-transformed centres) ---")
    centre_df = pd.DataFrame(centres_orig, columns=features)
    centre_df.index.name = "cluster"
    counts = labelled_df["cluster"].value_counts().sort_index()
    centre_df["n_countries"] = counts.values
    print(centre_df.round(2).to_string())

    print("\n--- Representative countries per cluster ---")
    for cl in sorted(labelled_df["cluster"].unique()):
        members = labelled_df[labelled_df["cluster"] == cl]["country_name"]
        sample = ", ".join(members.sample(
            min(5, len(members)), random_state=RANDOM_STATE
        ).tolist())
        print(f"  Cluster {cl} ({len(members)} countries): {sample}")


# ---------------------------------------------------------------------------
# Fitting model and error estimation
# ---------------------------------------------------------------------------

def logistic_model(x, L, k, x0):
    """
    Logistic (sigmoid) growth model: f(x) = L / (1 + exp(-k*(x - x0)))

    Captures saturation behaviour in CO2 trajectories — countries tend
    to plateau as they industrialise and adopt cleaner technologies.

    Parameters
    ----------
    x : array-like
        Independent variable (year).
    L : float
        Saturation level (upper asymptote).
    k : float
        Growth rate (steepness of the sigmoid).
    x0 : float
        Inflection year (midpoint of the sigmoid).

    Returns
    -------
    np.ndarray
        Model values.
    """
    return L / (1 + np.exp(-k * (x - x0)))


def fit_and_plot(ts_df, country_code=FIT_COUNTRY, pred_year=PRED_YEAR):
    """
    Fit an exponential model to CO2 per capita time series, estimate
    confidence intervals via error propagation, and produce a plot.

    Parameters
    ----------
    ts_df : pd.DataFrame
        Output of build_timeseries(), columns: year, co2_pc.
    country_code : str
        ISO code used in plot title.
    pred_year : int
        Year up to which predictions are shown.
    """
    years = ts_df["year"].values.astype(float)
    co2 = ts_df["co2_pc"].values

    # Reference year anchors the exponent to avoid overflow
    ref_year = years[0]

    # Initial parameter guesses for logistic: saturation, rate, midpoint
    p0 = [co2.max() * 1.3, 0.12, years.mean()]
    bounds = ([co2.max(), 0.01, years[0]],
              [co2.max() * 5, 1.0, years[-1]])

    popt, pcov = curve_fit(
        logistic_model,
        years,
        co2,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )

    print(f"\nFitted parameters (a, b, c): {np.round(popt, 4)}")
    print(f"Parameter uncertainties (1-sigma): "
          f"{np.round(np.sqrt(np.diag(pcov)), 4)}")

    # Extended x-axis for prediction
    x_pred = np.linspace(years[0], pred_year, 500)
    y_pred = logistic_model(x_pred, *popt)

    # Confidence interval via error propagation (from errors.py)
    sigma = errors.error_prop(x_pred, logistic_model, popt, pcov)
    upper = y_pred + sigma
    lower = y_pred - sigma

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(
        x_pred,
        lower,
        upper,
        alpha=0.25,
        color="#FF5722",
        label="1-sigma confidence range",
    )
    ax.plot(
        x_pred,
        y_pred,
        color="#FF5722",
        linewidth=2,
        label="Exponential fit",
    )
    ax.scatter(
        years,
        co2,
        color="#212121",
        s=40,
        zorder=4,
        label="World Bank observations",
    )

    # Mark prediction at target year
    y_at_pred = logistic_model(pred_year, *popt)
    sigma_at_pred = errors.error_prop(
        np.array([pred_year]), logistic_model, popt, pcov
    )[0]
    ax.errorbar(
        pred_year,
        y_at_pred,
        yerr=sigma_at_pred,
        fmt="D",
        color="#1565C0",
        markersize=8,
        capsize=6,
        label=f"{pred_year} prediction: "
              f"{y_at_pred:.2f} ± {sigma_at_pred:.2f} t/capita",
        zorder=5,
    )

    ax.axvline(years[-1], color="grey", linestyle="--", linewidth=1,
               alpha=0.6, label=f"Last observation ({int(years[-1])})")

    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("CO2 emissions per capita (metric tons)", fontsize=13)
    ax.set_title(
        f"CO2 per capita — {country_code}: exponential fit and "
        f"prediction to {pred_year}",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig("fitting_result.png", dpi=150)
    plt.show()
    print("  Saved: fitting_result.png")

    return popt, pcov, y_at_pred, sigma_at_pred


# ---------------------------------------------------------------------------
# Main programme
# ---------------------------------------------------------------------------

def main():
    """
    Main entry point.

    Execution order
    ---------------
    1. Download World Bank data.
    2. Run K-Means clustering and plot results.
    3. Fit exponential model to a country time series and plot results.
    """
    print("=" * 60)
    print("Assignment 3 — Clustering and Fitting")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Part 1: Clustering
    # ------------------------------------------------------------------
    print("\n[1/2] Running clustering analysis ...")

    snapshot = build_cluster_snapshot(year=CLUSTER_YEAR)
    labelled, centres_orig, scaler, features = run_clustering(
        snapshot, n_clusters=N_CLUSTERS
    )
    print_cluster_summary(labelled, centres_orig, features)
    plot_clusters(labelled, centres_orig, year=CLUSTER_YEAR)

    # ------------------------------------------------------------------
    # Part 2: Fitting
    # ------------------------------------------------------------------
    print("\n[2/2] Running fitting analysis ...")

    ts = build_timeseries(country_code=FIT_COUNTRY)
    popt, pcov, y_pred, sigma_pred = fit_and_plot(ts, country_code=FIT_COUNTRY)

    print(f"\nPredicted CO2 per capita in {PRED_YEAR}: "
          f"{y_pred:.2f} +/- {sigma_pred:.2f} metric tons")

    print("\nDone. Figures saved as clustering_result.png")
    print("and fitting_result.png.")


if __name__ == "__main__":
    main()
