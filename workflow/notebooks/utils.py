import json
import re
from io import BytesIO
from numbers import Integral, Real
from os import PathLike
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import scipy.stats
import sklearn.metrics
from coulson.data import HARTREE_TO_EV
from sklearn.linear_model import LinearRegression

AXIS_LABELS: dict[str, str] = {
    "t1_s1_ref": r"$\Delta E_{\mathrm{ST,REF}}$ (eV)",
    "t1_s1_ppp": r"$\Delta E_{\mathrm{ST,CIS}}$ (eV)",
    "t1_s1_dsp_scf": r"$\Delta E_{\mathrm{ST,SCF}}^{\mathrm{DSP}}$ (eV)",
    "t1_s1_dsp_cis": r"$\Delta E_{\mathrm{ST,CIS}}^{\mathrm{DSP}}$ (eV)",
    "t1_s1_dsp_cis_corr": r"$\Delta E_{\mathrm{ST,CIS,corr}}^{\mathrm{DSP}}$ (eV)",
    "exchange": r"$K$ (eV)",
    "dsp_scf": r"$\Delta E_{\mathrm{SCF}}^{\mathrm{DSP}}$ (eV)",
    "dsp_cis": r"$\Delta E_{\mathrm{CIS}}^{\mathrm{DSP}}$ (eV)",
    "homo_lumo_overlap": r"$O_\mathrm{HL}$",
    "oscillator_strength": r"$f_\mathrm{CIS}$",
    "oscillator_strength_ref": r"$f_\mathrm{REF}$",
    "s1_cis": r"$\Delta E_\mathrm{S,CIS}} (eV)$",
    "s1_ref": r"$\Delta E_\mathrm{S,REF}} (eV)$",
    "t1_cis": r"$\Delta E_\mathrm{T,CIS}} (eV)$",
    "t1_ref": r"$\Delta E_\mathrm{T,REF}} (eV)$",
    "t1_s1_cc2_aug_dz": r"$\Delta E_{\mathrm{ST,REF}}$ (eV)",
    "t1_s1_cc2_dz": r"$\Delta E_{\mathrm{ST,REF}}$ (eV)",
    "t1_s1_eom_aug_dz": r"$\Delta E_{\mathrm{ST,REF}}$ (eV)",
    "t1_s1_eom_dz": r"$\Delta E_{\mathrm{ST,REF}}$ (eV)",
}
"""Axis labels for plotting."""

SCAFFOLD_NAMES: dict[str, str] = {
    "I": "Cyclobuta-1,3-diene",
    "II": "Benzene",
    "III": "Cycloocta-1,3,5,7-tetraene",
    "IV": "Pentalene",
    "V": "Azulene",
    "VI": "Bowtiene",
    "VII": "Heptalene",
    "VIII": "Zurlene",
    "IX": "s-Indacene",
    "X": "as-Indacene",
    "XI": "Anthrazulene",
    "XII": "Phenazulene",
    "XIII": "Dicyclohepta[a,c]cyclobutene",
    "XIV": "Dicyclopenta[a,e]cyclooctene",
    "XV": "Dicyclopenta[a,d]cyclooctene",
    "XVI": "Dicyclopenta[a,c]cyclooctene",
}
"""Trivial names for the scaffolds."""

PARENT_SMILES: dict[str, str] = {
    "I": "C1=CC=C1",
    "II": "c1ccccc1",
    "III": "C1=CC=CC=CC=C1",
    "IV": "C1=CC2=CC=CC2=C1",
    "V": "c1ccc2cccc-2cc1",
    "VI": "c1cc2c3cccc-3c-2c1",
    "VII": "C1=CC=C2C=CC=CC=C2C=C1",
    "VIII": "C1=CC=C2C3=CC=CC3=C2C=C1",
    "IX": "C1=Cc2cc3c(cc2=C1)C=CC=3",
    "X": "C1=Cc2ccc3c(c2=C1)=CC=C3",
    "XI": "c1ccc2cc3cccc3cc2cc1",
    "XII": "c1ccc2ccc3cccc3c2cc1",
    "XIII": "c1ccc2c3cccncc-3c-2cc1",
    "XIV": "c1cc2ccc3cccc-3ccc-2c1",
    "XV": "c1cc2cccc-2cc2cccc-2c1",
    "XVI": "c1ccc2cccc-2c2cccc-2c1",
}
"""SMILES for the parent compounds of the rational design set."""

ALTERNANT: dict[str, str] = {
    "I": "alternant",
    "II": "alternant",
    "III": "alternant",
    "IV": "non-alternant",
    "V": "non-alternant",
    "VI": "non-alternant",
    "VII": "non-alternant",
    "VIII": "non-alternant",
    "IX": "non-alternant",
    "X": "non-alternant",
    "XI": "non-alternant",
    "XII": "non-alternant",
    "XIII": "non-alternant",
    "XIV": "non-alternant",
    "XV": "non-alternant",
    "XVI": "non-alternant",
}
"""Classification into alternant/non-alternant for the parent compounds of the rational design set."""


def plot_zero_zero(
    df: pd.DataFrame,
    x_name: str,
    y_name: str,
    plot_type: str = "scatter",
    fig: mpl.figure.Figure | None = None,
    ax: mpl.axes.Axes | None = None,
    zero_zero: bool = True,
    n_dec: int = 2,
    legend_loc: str | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes, dict[str, Real]]:
    """Plotting convenience function.

    Args:
        df: Dataframe
        x_name: Name of column for x axis
        y_name: Name of column for y axis
        plot_type: Type of plot: 'hexbin' or 'scatter' (default)
        fig: Matplotlib figure to use for plotting
        ax: Matplotlib axes to use for plotting
        zero_zero: Whether to do classification based on the zeros of x and  y
        n_dec: Number of decimals for regression coefficients in legend
        legend_loc: Location of legend in normal Matplotlib specification
        kwargs: Further plotting options forwarded to Pandas plotting function

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axes
        results: Dictionary of results for regression and classification

    """
    # Create format specifier for plotting
    format_specifier = f".{n_dec}f"

    # Create figure if it does not exist
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # Make plot
    allowed_values = ("scatter", "hexbin")
    if plot_type == "scatter":
        df.plot.scatter(x=x_name, y=y_name, ax=ax, **kwargs)
    elif plot_type == "hexbin":
        df.plot.hexbin(
            x=x_name,
            y=y_name,
            gridsize=100,
            bins="log",
            cmap="viridis",
            ax=ax,
            **kwargs,
        )
    else:
        raise ValueError(f"plot_type not in allowed values: {allowed_values}")

    # Set axes labels
    ax.set_xlabel(AXIS_LABELS[x_name])
    ax.set_ylabel(AXIS_LABELS[y_name])

    # Calculate regression scores
    y_true = df[y_name]
    lr = LinearRegression()
    lr.fit(df[[x_name]], df[y_name])
    y_pred = lr.predict(df[[x_name]])
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    spearman_r = scipy.stats.spearmanr(y_true, y_pred).statistic
    label = (
        rf"$R^2={r2:{format_specifier}}$"
        + "\n"
        + rf"$\rho={spearman_r:{format_specifier}}$"
    )

    results = {
        "$R^2$": r2,
        r"$\rho$": spearman_r,
    }

    # Calculate classification scores
    if zero_zero is True:
        ax.axhline(0.0, color="black", linestyle="--", zorder=0)
        ax.axvline(0.0, color="black", linestyle="--", zorder=0)
        y_pred = df[y_name] < 0
        y_true = df[x_name] < 0

        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
            y_true, y_pred, labels=[False, True]
        ).ravel()

        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = np.nan
        try:
            roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
        except ValueError:
            roc_auc = np.nan
        f1 = sklearn.metrics.f1_score(y_true, y_pred, zero_division=np.nan)
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred, zero_division=np.nan)

        label += "\n" + rf"F1$={f1:{format_specifier}}$"

        results.update(
            {
                "F1": f1,
                "ROC-AUC": roc_auc,
                "Accuracy": accuracy,
                "Recall": recall,
                "Specificity": specificity,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
            }
        )

    # Plot regression line
    x_min, x_max = ax.get_xlim()
    y_1, y_2 = [lr.coef_[0] * i + lr.intercept_ for i in (x_min, x_max)]
    ax.plot(
        [x_min, x_max],
        [y_1, y_2],
        linestyle="--",
        zorder=0,
        label=label,
    )

    # Add legend
    ax.legend(loc=legend_loc)

    return fig, ax, results


def load_data(filename: str | PathLike, filename_ref: str) -> pd.DataFrame:
    """Load calculated data from csv files.

    Args:
        filename: Filename or path of computed data
        filename_ref: Filename or path of reference data

    Returns:
        df_merged: Dataframe containing both computed and reference data.
    """
    # Load reference data
    df_ref = pd.read_csv(filename_ref, index_col=0)
    df_ref.columns = [
        "smiles",
        "s1_ref",
        "t1_ref",
        "t1_s1_ref",
        "oscillator_strength_ref",
    ]

    # Load computed data
    df = pd.read_csv(filename, index_col=0)
    df["t1_s1_ppp"] = df["s1_cis"] - df["t1_cis"]
    df["t1_s1_dsp_scf"] = (2 * df["exchange_integral"] + df["dsp_scf"]) * HARTREE_TO_EV
    df["t1_s1_dsp_cis"] = (df["dsp_cis"] + df["t1_s1_ppp"]) * HARTREE_TO_EV
    df["s1_cis"] *= HARTREE_TO_EV
    df["t1_cis"] *= HARTREE_TO_EV
    df["t1_s1_ppp"] *= HARTREE_TO_EV

    # Merge dataframes
    df_merged = pd.merge(df, df_ref, on="id")

    return df_merged


def load_garner(
    path_data: str | PathLike, path_ref: str | PathLike, path_lc: str | PathLike
) -> pd.DataFrame:
    """Load calculated Garner data from csv files.

    Args:
        path_data: Filename or path of computed data
        path_ref: Filename or path of reference data
        path_lc: Filename or path to the linear correction JSON

    Returns:
        df_merged: Dataframe containing both computed and reference data.
    """
    # Load computed data
    df_data = pd.read_csv(path_data, index_col=0)
    df_data["t1_s1_ppp"] = df_data["s1_cis"] - df_data["t1_cis"]
    df_data["t1_s1_dsp_scf"] = (
        2 * df_data["exchange_integral"] + df_data["dsp_scf"]
    ) * HARTREE_TO_EV
    df_data["t1_s1_dsp_cis"] = (
        df_data["dsp_cis"] + df_data["t1_s1_ppp"]
    ) * HARTREE_TO_EV
    df_data["s1_cis"] *= HARTREE_TO_EV
    df_data["t1_cis"] *= HARTREE_TO_EV
    df_data["t1_s1_ppp"] *= HARTREE_TO_EV

    # Load reference data
    df_ref = pd.read_csv(path_ref, index_col=0)
    df_ref.index.rename("id", inplace=True)
    df_ref.columns = [
        "t1_s1_cc2_aug_dz",
        "t1_s1_cc2_dz",
        "t1_s1_eom_aug_dz",
        "t1_s1_eom_dz",
    ]

    # Merge dataframes
    df = pd.merge(df_data, df_ref, on="id")

    # Add linear correction
    with open(path_lc) as f:
        d = json.load(f)
    intercept = d["Intercept"]
    coefficient = d["Coefficient"]

    df["t1_s1_dsp_cis_corr"] = df["t1_s1_dsp_cis"] * coefficient + intercept
    df["t1_s1_dsp_cis_corr"] = df["t1_s1_dsp_cis"] * coefficient + intercept
    df["instability"] = df["t1_cis"] < 0

    return df


def crop_image(png_bytes: bytes) -> bytes:
    """Crops PNG image.

    Follows:
    - https://gist.github.com/thomastweets/c7680e41ed88452d3c63401bb35116ed
    - https://stackoverflow.com/questions/33101935/convert-pil-image-to-byte-arra

    Args:
        png_bytes: PNG as bytes

    Returns:
        png_bytes_cropped: Cropped PNG as bytes
    """
    # Create PIL image from bytes
    image = PIL.Image.open(BytesIO(png_bytes))

    # Remove alpha channel
    image_rgb = image.convert("RGB")

    # Invert image (so that white is 0)
    image_inverted = PIL.ImageOps.invert(image_rgb)
    image_box = image_inverted.getbbox()

    # Crop image
    image_cropped = image.crop(image_box)

    # Convert back to bytes
    bytes_io = BytesIO()
    image_cropped.save(bytes_io, format="PNG")
    byte_string_cropped = bytes_io.getvalue()

    return byte_string_cropped


def format_dictionary_for_yaml(d: dict[str, Real], n_dec: int = 2) -> dict[str, str]:
    """Format dictionary with certain number of decimals for saving to YAML.

    Args:
        d: Dictionary to be formatted to string
        n_dec: Number of decimals for floating points

    Returns:
        d_formatted: Formatted dictionary
    """
    d_formatted = {}
    for key, value in d.items():
        if isinstance(value, Integral):
            value_formatted = str(value)
        elif isinstance(value, Real):
            value_formatted = f"{value:.{n_dec}f}"
        else:
            value_formatted = str(value)
        d_formatted[key] = value_formatted

    return d_formatted


def add_to_variables(
    variables: dict[str, Any],
    d: dict[str, Any],
    label_calculation: str,
    label_compound: str,
    label_method: str,
) -> dict[str, Any]:
    """Add dictionary to variables dictionary for later saving to Quarto.

    Args:
        variables: Variables dictionary that is added to
        d: Dictionary to add
        label_calculation: Label of calculation
        label_compound: Label of compound
        label_method: Label of method

    Returns:
        variables_new: Dictionary with all variables added
    """
    variables_new = dict(variables)
    for key, value in d.items():
        key_alnum = re.sub(r"[^A-Za-z0-9_]+", "", key)
        key_alnum_lower = key_alnum.lower()
        variables_new[
            f"{label_calculation}_{label_compound}_{key_alnum_lower}_{label_method}".strip(
                "_"
            )
        ] = value

    return variables_new
