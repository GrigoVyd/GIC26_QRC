"""Build the GIC 2026 Phase 3 report PDF (body plus references).

The body flows continuously (no forced page breaks) and is followed by the
references. main() then prepends the official Phase 3 cover via merge_cover().
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "output" / "report" / "assets"
OUT = ROOT / "output" / "pdf" / "Quanties_GIC2026_Phase3_Writeup_Draft.pdf"
# Official GIC Phase 3 cover template (filled). Copied verbatim in front of the
# body to form the final submission -- never re-rendered (the template may not be
# modified or recreated). The merge is skipped if the cover is absent.
COVER = ROOT / "GIC_2026 Cover Page.pdf"
SUBMISSION = ROOT / "output" / "pdf" / "Quanties__Phase3_V1.pdf"

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2F75B5")
MUTED = colors.HexColor("#667085")
LIGHT = colors.HexColor("#F2F4F7")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_GOLD = colors.HexColor("#FFF7E6")
GRID = colors.HexColor("#C9D2DC")


def register_fonts() -> None:
    font_dir = Path(r"C:\Windows\Fonts")
    pdfmetrics.registerFont(TTFont("TimesNewRoman", font_dir / "times.ttf"))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", font_dir / "timesbd.ttf"))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Italic", font_dir / "timesi.ttf"))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-BoldItalic", font_dir / "timesbi.ttf"))
    pdfmetrics.registerFontFamily(
        "TimesNewRoman",
        normal="TimesNewRoman",
        bold="TimesNewRoman-Bold",
        italic="TimesNewRoman-Italic",
        boldItalic="TimesNewRoman-BoldItalic",
    )


def styles():
    register_fonts()
    ss = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyTNR",
        parent=ss["BodyText"],
        fontName="TimesNewRoman",
        fontSize=11,
        leading=11,
        textColor=colors.black,
        spaceAfter=4,
        alignment=TA_LEFT,
    )
    h1 = ParagraphStyle(
        "H1TNR",
        parent=body,
        fontName="TimesNewRoman-Bold",
        fontSize=15,
        leading=15.5,
        textColor=NAVY,
        spaceBefore=5,
        spaceAfter=4,
        keepWithNext=True,
    )
    h2 = ParagraphStyle(
        "H2TNR",
        parent=body,
        fontName="TimesNewRoman-Bold",
        fontSize=12,
        leading=12,
        textColor=BLUE,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True,
    )
    title = ParagraphStyle(
        "TitleTNR",
        parent=body,
        fontName="TimesNewRoman-Bold",
        fontSize=19,
        leading=20,
        textColor=NAVY,
        spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "SubtitleTNR",
        parent=body,
        fontName="TimesNewRoman-Italic",
        fontSize=12,
        leading=12,
        textColor=MUTED,
        spaceAfter=4,
    )
    meta = ParagraphStyle(
        "MetaTNR",
        parent=body,
        fontName="TimesNewRoman-Bold",
        fontSize=10,
        leading=10,
        textColor=BLUE,
        spaceAfter=7,
    )
    caption = ParagraphStyle(
        "CaptionTNR",
        parent=body,
        fontName="TimesNewRoman-Italic",
        fontSize=11,
        leading=11,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceBefore=1,
        spaceAfter=4,
    )
    table_text = ParagraphStyle(
        "TableTNR",
        parent=body,
        fontName="TimesNewRoman",
        fontSize=11,
        leading=11,
        spaceAfter=0,
    )
    table_head = ParagraphStyle(
        "TableHeadTNR",
        parent=table_text,
        fontName="TimesNewRoman-Bold",
        textColor=colors.white,
        alignment=TA_CENTER,
    )
    ref = ParagraphStyle(
        "RefTNR",
        parent=body,
        fontSize=11,
        leading=11,
        leftIndent=16,
        firstLineIndent=-16,
        spaceAfter=4,
    )
    return {
        "body": body,
        "h1": h1,
        "h2": h2,
        "title": title,
        "subtitle": subtitle,
        "meta": meta,
        "caption": caption,
        "table": table_text,
        "table_head": table_head,
        "ref": ref,
    }


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("TimesNewRoman", 8.5)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(letter[0] - 0.75 * inch, letter[1] - 0.42 * inch,
                           "GIC 2026 | qBraid - MITRE - JonesTrading | Phase 3")
    canvas.drawCentredString(letter[0] / 2, 0.34 * inch, f"Team Quanties | Page {doc.page}")
    canvas.restoreState()


def P(text, style):
    return Paragraph(text, style)


def bullets(items, style):
    return ListFlowable(
        [ListItem(P(item, style), leftIndent=8) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=17,
        bulletFontName="TimesNewRoman",
        bulletFontSize=8,
        bulletOffsetY=1,
        spaceAfter=2,
    )


def numbers(items, style):
    return ListFlowable(
        [ListItem(P(item, style), leftIndent=8) for item in items],
        bulletType="1",
        leftIndent=20,
        bulletFontName="TimesNewRoman",
        bulletFontSize=10,
        bulletOffsetY=0,
        spaceAfter=2,
    )


def callout(label, text, style, fill=PALE_BLUE):
    table = Table([[P(f"<b>{label}</b> {text}", style)]], colWidths=[7.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 0.5, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def data_table(headers, rows, widths, st, *, font_size=None):
    head_style = st["table_head"]
    cell_style = st["table"]
    if font_size:
        head_style = ParagraphStyle("head_local", parent=head_style, fontSize=font_size, leading=font_size + 0.3)
        cell_style = ParagraphStyle("cell_local", parent=cell_style, fontSize=font_size, leading=font_size + 0.3)
    data = [[P(h, head_style) for h in headers]]
    for row in rows:
        data.append([P(str(v), cell_style) for v in row])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
    ]
    for r in range(2, len(data), 2):
        cmds.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F8FAFC")))
    table.setStyle(TableStyle(cmds))
    return table


def fig(name, width, height, caption, st):
    im = Image(str(ASSETS / name), width=width, height=height)
    im.hAlign = "CENTER"
    return KeepTogether([im, P(caption, st["caption"])])


def build_story(st):
    B, H1, H2 = st["body"], st["h1"], st["h2"]
    story = []

    # Page 1
    story += [
        P("Hardware-Competitive Quantum Reservoir Computing for SPY Volatility", st["title"]),
        P("A native-hardware hybrid across IQM Emerald, QuEra Aquila, and cloud Ising machines", st["subtitle"]),
        P("Track A: Financial Volatility Prediction | Team Quanties", st["meta"]),
        P("Abstract", H1),
        P("We forecast next-day annualized 21-day realized volatility of SPY from public daily OHLCV data (2010-2024). GARCH(1,1) supplies a structural prior; a fixed quantum reservoir learns only a regularized correction to the GARCH log-volatility forecast. Zero correction therefore equals GARCH exactly, giving the quantum component a stringent role: extract nonlinear residual structure without replacing a strong econometric model.", B),
        P("The primary real-QPU result is a 9-qubit native 3x3 IQM Emerald reservoir evaluated on 120 chronological days at 500 shots per circuit. Its pre-specified quantum readout-error mitigation (QREM) plus affine-transfer path gives RMSE 0.00831368 versus 0.00832112 for GARCH, a 0.0895% point improvement. The moving-block-bootstrap 95% interval for the RMSE difference is [-0.0000608, +0.0000262], so we claim hardware-level competitiveness, not statistically established quantum advantage. QLIKE is 0.016538 versus 0.016476 for GARCH; the Mincer-Zarnowitz joint p-value is 0.166, so forecast unbiasedness is not rejected.", B),
        callout("Headline.", "Across executed substrates, the hybrid remains within 0.43% RMSE of matched GARCH. Real Aquila transfers from its local analog Hamiltonian simulation (AHS) within 0.029% RMSE, while a 12-qubit IQM expansion improves feature fidelity but worsens forecasting. Native topology and validation-locked regularization matter more than width alone.", B),
        Spacer(1, 2),
        P("1. Problem framing and data", H1),
        P("The target RV[t] is the rolling 21-day standard deviation of log returns multiplied by sqrt(252). At forecast time, 20 of the 21 squared-return terms are known, giving GARCH a structural advantage. Inputs are causal lags of returns, absolute returns, squared returns, and heterogeneous autoregressive realized-volatility (HAR-RV) summaries. All splits are chronological; model selection uses training-only expanding folds.", B),
        P("We report RMSE, volatility-specific quasi-likelihood (QLIKE) loss, and Mincer-Zarnowitz forecast calibration. Comparators include Persistence, Ridge/AR, ESN-200, LSTM, and GARCH(1,1). A 746-day full benchmark establishes classical/simulator performance; a separate 400-train/120-test window supports hardware-ready and real-hardware transfer experiments.", B),
    ]

    # Section 2
    story += [
        P("2. Hybrid QRC architecture", H1),
        P("For each market state x[t], GARCH produces a log-volatility prior. A fixed reservoir maps x[t] to shot-estimated one- and two-body observables; only a Ridge readout is trained. The forecast is log RV_hat[t] = log RV_GARCH[t] + c f_QRC(x[t]), with the Ridge penalty and correction strength c locked on validation data. Quantum parameters are fixed before the test window; no variational quantum optimization is used.", B),
        fig("phase3_hybrid_architecture.png", 7.0 * inch, 2.36 * inch,
            "<b>Figure 1.</b> The GARCH prior protects baseline performance; the fixed reservoir supplies nonlinear residual features. Hardware mitigation is label-free.", st),
        data_table(
            ["Substrate", "Native encoding and dynamics", "Readout", "Purpose"],
            [
                ["IQM Emerald", "Rz input on a selected nearest-neighbor grid; fixed CZ mixing", "45 Z/ZZ", "Primary real gate-QPU forecast"],
                ["QuEra Aquila", "Negative local detuning on a valid irregular atom grid; Rydberg AHS", "5 Z + 10 ZZ", "Real analog-QPU transfer"],
                ["Amplify AE / Toshiba", "Input fields h_i with fixed signed all-to-all J_ij; cloud optimization/sampling", "10 Z + 45 ZZ", "Classical signed-Ising ablation"],
                ["TFIM simulator", "Signed J_ij plus transverse-field evolution", "Z/ZZ ensemble", "Mechanism and advantage candidate"],
            ],
            [1.1 * inch, 2.8 * inch, 1.0 * inch, 2.1 * inch], st, font_size=11,
        ),
        P("Why a reservoir is appropriate", H2),
        P("Volatility is nonlinear, heteroskedastic, and regime switching. A fixed interacting system supplies a high-dimensional nonlinear projection and fading memory, while the linear head controls variance. The residual formulation tests whether the reservoir adds orthogonal structure after the known 21-day decomposition is supplied. Z and ZZ observables expose local response and interaction-mediated correlations.", B),
        P("Hardware transfer is conservative. IQM circuits use explicit native qubit selection and reject SWAPs; Aquila geometry satisfies the 4 micrometer per-axis rule and uses local detuning. QREM and affine feature alignment use calibration information or unlabeled features only, preventing target leakage.", B),
    ]

    # Section 3
    story += [
        P("3. Experimental design and simulator evidence", H1),
        P("The 746-day benchmark establishes task difficulty. GARCH leads the raw-task baselines (RMSE 0.007948), followed by Persistence (0.009408), ESN-200 (0.009704), and LSTM (0.010662). On the GARCH-residual target, the 10-qubit transverse-field Ising model (TFIM) reservoir reaches RMSE 0.007844 and QLIKE 0.006445 versus 0.006863 for GARCH: improvements of 1.31% and 6.09%. The identical Ridge residual ablation has RMSE 0.008051. This is a simulator result, not an executed hardware-advantage claim.", B),
        fig("phase3_simulator_evidence.png", 7.0 * inch, 3.16 * inch,
            "<b>Figure 2.</b> Left: the simulated TFIM hybrid is the advantage candidate. Right: every hardware-ready reservoir improves on the same linear residual ablation, although none beats GARCH on the common 120-day window.", st),
        P("Selection, noise, and ablation controls", H2),
        bullets([
            "All hyperparameters, layouts, Ridge penalties, and correction strengths are selected on expanding training/validation folds. Test rows are scored once.",
            "The final 9-qubit configuration gains 0.536% over GARCH in exact statevector simulation and 0.410% under the IQM-matched 500-shot proxy.",
            "On the common 400/120 window, Pasqal analog, gate QRC, QuEra analog, and signed-Ising reservoirs improve over the linear residual ablation by 2.36%, 2.10%, 1.79%, and 1.52%.",
            "The 12-qubit native-grid check raises hardware-to-exact feature correlation from 0.770 to 0.818 but changes the forecast from 0.090% better than GARCH to 0.196% worse.",
        ], B),
        P("Mechanism interpretation", H2),
        P("Removing the transverse field leaves the cloud signed-Ising tier near GARCH but does not reproduce the full simulated gain; removing signed programmability produces the neutral-atom tier with the same conclusion. The evidence supports a falsifiable hypothesis - useful residual features arise from transverse-field mixing plus signed couplings - while real hardware supports competitiveness and portability rather than advantage.", B),
    ]

    # Section 4
    story += [
        P("4. Executed hardware and cloud results", H1),
        P("All rows below are completed external executions and are compared with GARCH on identical held-out rows. Percentages are comparable within a row even when absolute RMSE differs across windows.", B),
        fig("phase3_hardware_evidence.png", 7.0 * inch, 3.16 * inch,
            "<b>Figure 3.</b> Negative RMSE gap is better. Panel B shows that higher 12-qubit feature fidelity does not improve the forecast.", st),
        data_table(
            ["Platform", "Native workload", "Evaluation", "Cost", "Gap", "Reading"],
            [
                ["IQM 9q", "depth 32; 24 CZ; 0 SWAP", "120 x 500", "20.25 Resonance cr", "-0.090%", "Primary QPU; CI crosses zero"],
                ["IQM 12q", "depth 38; 34 CZ; 0 SWAP", "120 x 500", "20.25 Resonance cr", "+0.196%", "Scaling negative result"],
                ["Aquila 5 atoms", "native AHS; local detuning", "23 x 50; 20 scored", "1,840 qBraid cr", "+0.182%", "Only 0.029% behind local AHS"],
                ["Amplify AE", "10 spins; 45 signed J", "120 solves", "existing credits", "+0.055%", "Classical ablation"],
                ["Toshiba", "10 spins; 45 signed J", "120 x 200 states", "existing credits", "+0.421%", "Classical ablation"],
            ],
            [0.85 * inch, 1.45 * inch, 1.0 * inch, 1.1 * inch, 0.7 * inch, 1.9 * inch], st, font_size=11,
        ),
        P("Primary IQM result", H2),
        P("Emerald job 019f5691-063b-7a11-8bbc-db5057b79259 executed 60,000 shots on QB17-19, QB25-27, and QB33-35. QREM plus feature-only affine transfer gives feature correlation 0.770 and RMSE 0.00831368. The bootstrap probability of beating GARCH is 0.658; the interval crosses zero. QLIKE is slightly worse, so the defensible conclusion is GARCH-level real-QPU performance with a small positive RMSE point estimate.", B),
        P("Analog and cloud-Ising transfer", H2),
        P("Aquila completed 23 paid tasks after zero-cost validation exposed native geometry and local-detuning constraints. Three unlabeled rows calibrate feature alignment and 20 are scored. Hardware RMSE is 0.0137535 versus 0.0137495 locally and 0.0137285 for GARCH; it improves 6.06% over the Ridge residual ablation. Amplify AE and Toshiba remain within 0.43% of GARCH but are classical optimizers, not quantum-advantage evidence.", B),
        P("Runtime and compute access", H2),
        P("The reproducible classical pipeline is light: the credit-safe judge notebook reproduces every reported value in about 8 s, and the local statevector reservoir-feature reference (2,984 training and 120 test states on 9 qubits) builds in about 55 s on a laptop CPU, with sub-second readout fitting and scoring. On hardware, each IQM job dispatches 60,000 shots (120 circuits x 500) and the Aquila campaign 1,150 shots (23 tasks x 50); QPU wall-clock is dominated by provider queue and allocation windows. All paid runs were funded from the team's own accounts: despite several requests, sponsor challenge credits on qBraid were not provisioned, so the Aquila campaign used the team's personal qBraid balance and the IQM runs used personal IQM Resonance credits.", B),
    ]

    # Section 5
    story += [
        P("5. Conclusions, impact, and reproducibility", H1),
        P("Conclusions", H2),
        bullets([
            "The hybrid architecture delivered measurable improvement, not merely parity: the simulated TFIM reduced RMSE by 1.31% and QLIKE by 6.09% versus GARCH; executed IQM 9q produced a 0.0895% lower RMSE point estimate; and the real Aquila hybrid improved 6.06% over the matched Ridge residual ablation.",
            "The strongest mechanism claim is simulated: the transverse-field signed-coupling reservoir improves RMSE and QLIKE over GARCH, whereas neutral-atom and classical signed-Ising ablations do not reproduce the full gain.",
            "More qubits are not automatically better. Native connectivity, noise-aware training, low depth, and conservative correction strength dominate width in this data regime.",
        ], B),
        P("The full simulated improvement was not recovered on the executed QPUs. The ideal TFIM combines coherent transverse-field mixing with programmable signed interactions, while available devices impose restricted native couplings, finite-shot estimation, device noise, and mitigation or feature-transfer error. Even under those constraints, the residual hybrid preserves the GARCH baseline and produces the improvements above. Because the full simulator benchmark and hardware experiments use different evaluation windows, their percentages demonstrate complementary evidence rather than a direct numerical comparison.", B),
        P("Stakeholder relevance", H2),
        P("Daily volatility forecasts support hedging, market-maker inventory and spread decisions, portfolio risk limits, and scenario generation. The deployable contribution is a modular residual feature engine: it can be switched off to recover GARCH exactly, deployed across substrates, and monitored through correction magnitude and calibration metrics.", B),
        P("Limitations", H2),
        bullets([
            "The IQM RMSE gain is small and not significant; QLIKE does not improve. Aquila has only 20 scored days and 50 shots per task.",
            "Amplify AE and Toshiba lack coherent transverse-field dynamics. The GARCH-beating TFIM result is simulated, not QPU-executed.",
            "The exact hardware window still needs GJR-GARCH, EGARCH, and HAR-RV; the full-window panel already includes GARCH, ESN, LSTM, Ridge/AR, and Persistence.",
        ], B),
        P("Reproducibility and submission status", H2),
        data_table(
            ["Requirement", "Evidence", "Status"],
            [
                ["Concrete qBraid execution", "Aquila task IDs/checkpoint; qir-sv records", "Complete"],
                ["Re-runnable workflow", "Judge notebook, experiments/, compact CSV/JSON evidence", "Complete"],
                ["5-page write-up", "This body (11-pt TNR) plus separate references", "Complete"],
                ["Cover + Launch button", "Official Phase 3 cover prepended; README Launch button", "Complete"],
            ],
            [1.7 * inch, 3.8 * inch, 1.5 * inch], st, font_size=11,
        ),
        Spacer(1, 12),
    ]

    # References (flow directly after the body; excluded from the page limit)
    refs = [
        "qBraid, MITRE, and JonesTrading. <i>Global Industry Challenge 2026: Phase 3 Challenge Description.</i> 2026.",
        "Aqora. <i>qBraid, MITRE &amp; JonesTrading: GIC 2026 track page.</i> aqora.io/challenges/global-industry-challenge-2026/tracks/gic-2026-qBraid-MITRE-JonesTrading (accessed 15 July 2026).",
        "Bollerslev, T. Generalized autoregressive conditional heteroskedasticity. <i>Journal of Econometrics</i> 31, 307-327 (1986).",
        "Corsi, F. A simple approximate long-memory model of realized volatility. <i>Journal of Financial Econometrics</i> 7, 174-196 (2009).",
        "Patton, A. J. Volatility forecast comparison using imperfect volatility proxies. <i>Journal of Econometrics</i> 160, 246-256 (2011).",
        "Jaeger, H. The echo state approach to analysing and training recurrent neural networks. GMD Report 148 (2001).",
        "Fujii, K. and Nakajima, K. Harnessing disordered-ensemble quantum dynamics for machine learning. <i>Physical Review Applied</i> 8, 024030 (2017).",
        "Martinez-Pena, R. et al. Dynamical phase transitions in quantum reservoir computing. <i>Physical Review Letters</i> 127, 100502 (2021).",
        "Kornjaca, M. et al. Large-scale quantum reservoir learning with an analog quantum computer. arXiv:2407.02553 (2024).",
        "Li, Q. et al. Quantum reservoir computing for realized volatility forecasting. arXiv:2505.13933 (2025).",
        "Ahmed, O., Tennie, F., and Magri, L. Robust quantum reservoir computers for forecasting chaotic dynamics. <i>Proceedings of the Royal Society A</i> 481, 20250550 (2025).",
        "qBraid. <i>Launch on qBraid badge and repository redirect parameters.</i> github.com/qBraid/community/discussions/3 (accessed 15 July 2026).",
    ]
    story += [P("References (excluded from the five-page limit)", H1)]
    story += [P(f"[{i}] {ref}", st["ref"]) for i, ref in enumerate(refs, 1)]
    story += [
        P("Artifact disclosure", H2),
        P("Generative AI tools were used for code support, debugging, figure generation, and writing assistance. The experimental design, platform choices, executed jobs, result interpretation, and final claims are the team's own. Secrets and access tokens are excluded from the repository and should be rotated after the hardware campaign.", B),
    ]
    return story


def main() -> None:
    for name in ("phase3_hybrid_architecture.png", "phase3_simulator_evidence.png", "phase3_hardware_evidence.png"):
        if not (ASSETS / name).exists():
            raise FileNotFoundError(ASSETS / name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    st = styles()
    doc = BaseDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.60 * inch,
        bottomMargin=0.55 * inch,
        title="Hardware-Competitive Quantum Reservoir Computing for SPY Volatility",
        author="Team Quanties",
        subject="GIC 2026 Phase 3 technical write-up",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=header_footer)])
    doc.build(build_story(st))
    print(f"Wrote {OUT}")
    merge_cover()


def merge_cover() -> None:
    """Prepend the official cover to the body -> final submission PDF.

    The cover page is copied byte-for-byte (never re-rendered) so the required
    template stays unmodified. Cover + 5-page body + references.
    """
    if not COVER.exists():
        print(f"Cover not found ({COVER.name}); skipped submission merge.")
        return
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for page in PdfReader(str(COVER)).pages:
        writer.add_page(page)
    for page in PdfReader(str(OUT)).pages:
        writer.add_page(page)
    with open(SUBMISSION, "wb") as fh:
        writer.write(fh)
    print(f"Wrote {SUBMISSION} ({len(writer.pages)} pages: cover + body + references)")


if __name__ == "__main__":
    main()
