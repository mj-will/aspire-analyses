"""Convert JSON files with evidence estimates into a LaTeX table."""
from pathlib import Path
import json


def fmt(x, ndp=2):
    if x is None or x == "":
        return ""
    return f"{x:.{ndp}f}"


def fmt_int(x):
    if x is None or x == "":
        return ""
    return f"{int(x):,}"


def fmt_ratio(n, d, ndp=2):
    if n in (None, "") or d in (None, "", 0):
        return ""
    return f"{n / d:.{ndp}f}"


def highlight(logz, sigma, analytic, ndp=2):
    """Bold logZ if within 3σ of analytic."""
    value = fmt(logz, ndp=ndp)
    if sigma in (None, ""):
        return value
    if abs(logz - analytic) <= (3 * sigma):
        return rf"\textbf{{{value}}}"
    return value


def ptmcmc_block(label, keys, evidence_results, analytic_value):
    """Generate rows in the CORRECT column order."""
    ti_logz, ti_err = evidence_results[keys["ti"]]
    _, ti_coarse_err = evidence_results[keys["ti_coarse"]]
    ss_logz, ss_err = evidence_results[keys["ss"]]

    n_samples = evidence_results[keys["n_samples"]]
    n_evals = evidence_results[keys["evals"]]

    return [
        (
            label,
            fmt_int(n_samples),
            fmt_ratio(n_evals, n_samples),
            "Thermodynamic Integration",
            highlight(ti_logz, ti_err, analytic_value),
            f"{fmt(ti_err)} ({fmt(ti_coarse_err)})",
        ),
        (
            "",
            "",
            "",
            "Stepping Stone",
            highlight(ss_logz, ss_err, analytic_value),
            fmt(ss_err),
        ),
    ]


def main():
    json_file = (
        Path("outdir")
        / "gaussian_comparison_fixed"
        / "15d"
        / "evidence_results.json"
    )

    output = Path("evidence_tables")
    output.mkdir(parents=True, exist_ok=True)

    with open(json_file, "r") as f:
        evidence_results = json.load(f)

    analytic_value = evidence_results["log_z"]

    # --- Rows in correct column order ---
    analytic_row = (
        "Analytic",
        "",
        "",
        "",
        fmt(analytic_value),
        "",
    )

    smc_samples = evidence_results["smc_n_samples"]
    smc_evals = evidence_results["smc_likelihood_evals"]
    smc_logz, smc_err = evidence_results["smc_log_z"]

    smc_row = (
        "ASPIRE SMC",
        fmt_int(smc_samples),
        fmt_ratio(smc_evals, smc_samples),
        r"\Cref{eq:smc:Z_estimator}",
        highlight(smc_logz, smc_err, analytic_value),
        fmt(smc_err),
    )

    # --- Ordered PTMCMC definitions ---
    ptmcmc_methods = [
        (
            "PTMCMC A",
            {
                "ti": "ptmcmc_log_z_ti",
                "ti_coarse": "ptmcmc_log_z_ti_coarse",
                "ss": "ptmcmc_log_z_ss",
                "n_samples": "ptmcmc_n_samples",
                "evals": "ptmcmc_likelihood_evals",
            },
        ),
        (
            "PTMCMC B",
            {
                "ti": "ptmcmc_alt_log_z_ti",
                "ti_coarse": "ptmcmc_alt_log_z_ti_coarse",
                "ss": "ptmcmc_alt_log_z_ss",
                "n_samples": "ptmcmc_alt_n_samples",
                "evals": "ptmcmc_alt_likelihood_evals",
            },
        ),
        (
            "PTMCMC C",
            {
                "ti": "ptmcmc_flow_log_z_ti",
                "ti_coarse": "ptmcmc_flow_log_z_ti_coarse",
                "ss": "ptmcmc_flow_log_z_ss",
                "n_samples": "ptmcmc_flow_n_samples",
                "evals": "ptmcmc_flow_likelihood_evals",
            },
        ),
    ]

    ptmcmc_rows = []
    for label, keys in ptmcmc_methods:
        ptmcmc_rows.extend(
            ptmcmc_block(label, keys, evidence_results, analytic_value)
        )

    # --- LaTeX (column order matches exactly) ---
    latex_lines = [
        r"\begin{tabular}{p{2.5cm} C{2cm} C{2.5cm} C{2.5cm} c c}",
        r"\toprule",
        r"Method & Posterior samples & Likelihood evals. per sample & Estimator & Log Evidence & Uncertainty \\",
        r"\midrule",
        rf"{analytic_row[0]} & {analytic_row[1]} & {analytic_row[2]} & {analytic_row[3]} & {analytic_row[4]} & {analytic_row[5]} \\",
        "\midrule",
        rf"{smc_row[0]} & {smc_row[1]} & {smc_row[2]} & {smc_row[3]} & {smc_row[4]} & {smc_row[5]} \\",
    ]

    for i in range(0, len(ptmcmc_rows), 2):
        method, ns1, cps1, est1, logz1, err1 = ptmcmc_rows[i]
        _, ns2, cps2, est2, logz2, err2 = ptmcmc_rows[i + 1]

        latex_lines.append(r"\midrule")
        latex_lines.append(rf"\multirow{{2}}{{*}}{{{method}}}")
        latex_lines.append(
            rf" & {ns1} & {cps1} & {est1} & {logz1} & {err1} \\"
        )
        latex_lines.append(
            rf" & {ns2} & {cps2} & {est2} & {logz2} & {err2} \\"
        )

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    latex = "\n".join(latex_lines) + "\n"

    with open(output / "evidence_table_gaussian.tex", "w") as f:
        f.write(latex)


if __name__ == "__main__":
    main()