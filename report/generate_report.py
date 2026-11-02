"""Generate the academic project PDF from real run artifacts."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Image,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    Frame,
    PageBreak,
)
from reportlab.platypus.tableofcontents import TableOfContents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
SCREENSHOT_CHECKLIST = PROJECT_ROOT / "screenshots" / "CHECKLIST.md"


class ProjectReportTemplate(BaseDocTemplate):
    def __init__(self, filename: str | Path, styles: dict[str, ParagraphStyle]) -> None:
        super().__init__(
            str(filename),
            pagesize=letter,
            rightMargin=0.72 * inch,
            leftMargin=0.72 * inch,
            topMargin=0.78 * inch,
            bottomMargin=0.68 * inch,
            title="Predicting Customer Purchase Behaviour Using Apache Spark and Machine Learning",
            author="INT315 - Cluster Computing (CA2)",
        )
        self._styles = styles
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates(
            [PageTemplate(id="main", frames=frame, onPage=self._draw_page_number)]
        )
        self._heading_count = 0

    def beforeDocument(self) -> None:
        self._heading_count = 0

    @staticmethod
    def _draw_page_number(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5B6573"))
        canvas.drawCentredString(letter[0] / 2, 0.38 * inch, str(document.page))
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name == "Heading1":
            self._heading_count += 1
            title = flowable.getPlainText()
            key = f"section-{self._heading_count}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (0, title, self.page, key))


def _styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitlePage",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=29,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=18,
        )
    )
    styles["Heading1"].fontSize = 15
    styles["Heading1"].leading = 19
    styles["Heading1"].textColor = colors.HexColor("#17365D")
    styles["Heading1"].spaceBefore = 13
    styles["Heading1"].spaceAfter = 8
    styles["Heading2"].fontSize = 11
    styles["Heading2"].leading = 14
    styles["Heading2"].textColor = colors.HexColor("#2F5597")
    styles["Heading2"].spaceBefore = 9
    styles["Heading2"].spaceAfter = 5
    styles.add(
        ParagraphStyle(
            name="BodyAcademic",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=14,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeSmall",
            fontName="Courier",
            fontSize=5.8,
            leading=7.1,
            leftIndent=5,
            rightIndent=5,
            backColor=colors.HexColor("#F3F5F7"),
        )
    )
    styles.add(ParagraphStyle(name="TOCHeading", parent=styles["Heading1"]))
    styles.add(
        ParagraphStyle(
            name="TOCEntry0",
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            leftIndent=8,
            firstLineIndent=-8,
            spaceBefore=3,
        )
    )
    return styles


def _paragraph(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text, styles["BodyAcademic"])


def _section(title: str, styles: dict[str, ParagraphStyle], story: list) -> None:
    story.append(Paragraph(title, styles["Heading1"]))


def _screenshot_placeholder(label: str) -> Table:
    table = Table(
        [[Paragraph(f"<b>Screenshot placeholder:</b> {label}<br/>Insert an authentic project screenshot here.", getSampleStyleSheet()["BodyText"])]],
        colWidths=[6.7 * inch],
        rowHeights=[0.72 * inch],
    )
    table.setStyle(
        [
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#7F8C9A")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F6F8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]
    )
    return table


def _metric_table(evaluation: dict[str, Any]) -> Table:
    rows = [["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]]
    for model_name, metrics in evaluation["models"].items():
        rows.append(
            [
                model_name,
                *[f"{metrics[name]:.4f}" for name in ("accuracy", "precision", "recall", "f1", "roc_auc")],
            ]
        )
    table = Table(rows, repeatRows=1, colWidths=[1.55 * inch, 1.0 * inch, 0.9 * inch, 0.8 * inch, 0.8 * inch, 0.9 * inch])
    table.setStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#ABB7C4")),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F8")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
    )
    return table


def _source_files() -> list[Path]:
    paths = [PROJECT_ROOT / "main.py", *sorted((PROJECT_ROOT / "src").glob("*.py"))]
    paths.extend([PROJECT_ROOT / "dashboard" / "app.py", Path(__file__).resolve()])
    return [path for path in paths if path.is_file()]


def _appendix_code(path: Path, styles: dict[str, ParagraphStyle], story: list) -> None:
    relative_path = path.relative_to(PROJECT_ROOT)
    story.append(Paragraph(str(relative_path), styles["Heading2"]))
    wrapped_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=112,
                subsequent_indent="    ",
                replace_whitespace=False,
                drop_whitespace=False,
            )
            or [""]
        )
    story.append(Preformatted("\n".join(wrapped_lines), styles["CodeSmall"]))
    story.append(Spacer(1, 8))


def generate_report(
    preprocessing_summary: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Create a report using actual run metrics; refuse missing execution results."""
    summary_path = OUTPUTS / "preprocessing_summary.json"
    metrics_path = OUTPUTS / "model_metrics.json"
    if preprocessing_summary is None:
        if not summary_path.is_file():
            raise FileNotFoundError("Run main.py before generating the report.")
        preprocessing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if evaluation is None:
        if not metrics_path.is_file():
            raise FileNotFoundError("Model metrics are missing; run main.py first.")
        evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not evaluation.get("models"):
        raise ValueError("The report requires actual metrics for at least one trained model")

    destination = Path(output_path) if output_path else PROJECT_ROOT / "report" / "project_report.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    story: list[Any] = [
        Spacer(1, 1.2 * inch),
        Paragraph(
            "Predicting Customer Purchase Behaviour<br/>Using Apache Spark and Machine Learning",
            styles["TitlePage"],
        ),
        Spacer(1, 0.35 * inch),
        _paragraph("INT315 - Cluster Computing (CA2)", styles),
        Spacer(1, 0.2 * inch),
        _paragraph("Academic project report", styles),
        Spacer(1, 2.4 * inch),
        _paragraph(
            "Prepared with measured outputs from the local Spark execution. "
            "No unrun results or screenshots are represented as evidence.",
            styles,
        ),
        PageBreak(),
        Paragraph("Table of Contents", styles["TOCHeading"]),
    ]
    toc = TableOfContents()
    toc.levelStyles = [styles["TOCEntry0"]]
    story.extend([toc, PageBreak()])

    _section("1. Project Title", styles, story)
    story.append(_paragraph("Predicting Customer Purchase Behaviour Using Apache Spark and Machine Learning.", styles))

    _section("2. Problem Statement", styles, story)
    story.append(
        _paragraph(
            "E-commerce teams need to understand which browsing sessions are associated with a purchase. "
            "This project uses session-level website interaction data to classify whether a session ends "
            "with a transaction. The target is binary: 1 indicates that a purchase was recorded and 0 "
            "indicates no purchase.",
            styles,
        )
    )

    _section("3. Objective of the Project", styles, story)
    story.append(
        _paragraph(
            "The objective is to build a reproducible Spark workflow that loads and validates the public "
            "dataset, performs leakage-aware preprocessing, trains Logistic Regression and Gradient-Boosted "
            "Trees, evaluates the models on a held-out test set, exports session predictions, and presents "
            "the measured outputs in a dashboard.",
            styles,
        )
    )

    _section("4. Dataset Description", styles, story)
    story.append(
        _paragraph(
            "The dataset contains online-shopping sessions rather than demographic customer profiles. "
            "The public UCI record describes 12,330 sessions, with one session per user in the sampled "
            "period. The project keeps the source features and uses a generated session identifier only "
            "for output traceability; that identifier is not a source customer ID and is excluded from training. "
            f"The local project copy used for this run is {Path(dataset_path).name if dataset_path else 'customer_behaviour.csv'}.",
            styles,
        )
    )
    column_rows = [
        ["Field(s)", "Meaning"],
        ["Administrative, Informational, ProductRelated", "Counts of visited pages in each page category."],
        ["*_Duration", "Time spent in each corresponding page category during the session."],
        ["BounceRates, ExitRates, PageValues", "Web analytics measures associated with the session."],
        ["SpecialDay", "Proximity measure for a special shopping day."],
        ["Month, OperatingSystems, Browser, Region, TrafficType", "Session time and technical/traffic categories."],
        ["VisitorType, Weekend", "New/returning visitor category and weekend indicator."],
        [        "Revenue -> purchase", "Binary outcome; True maps to 1 (purchase), False to 0."],
    ]
    schema_table = Table(column_rows, repeatRows=1, colWidths=[2.5 * inch, 4.2 * inch])
    schema_table.setStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#ABB7C4")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
    story.append(schema_table)
    story.append(
        _paragraph(
            "Limitations: the source has no age/age-group, explicit session-duration field, or previous "
            "purchase-count field. Session duration in this project is derived from the sum of the three "
            "page-category duration features. VisitorType distinguishes new from returning visitors but "
            "is not a count of past purchases. The project does not fabricate the missing attributes.",
            styles,
        )
    )

    _section("5. Dataset Source", styles, story)
    story.append(
        _paragraph(
            "Source: Sakar, C. and Kastro, Y. (2018), Online Shoppers Purchasing Intention Dataset, "
            "UCI Machine Learning Repository. DOI: 10.24432/C5F88Q. "
            "Dataset page: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset. "
            "Direct archive: https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip. "
            "The dataset is distributed under CC BY 4.0; attribution is required.",
            styles,
        )
    )

    _section("6. Tools and Technologies Used", styles, story)
    story.append(
        _paragraph(
            "Python 3, Apache Spark in local[*] mode, PySpark SQL and MLlib, Pandas, NumPy, "
            "Matplotlib, Seaborn, scikit-learn for ROC curve coordinates, Streamlit, Plotly, "
            "and ReportLab. Local mode uses one machine and is not represented as a multi-node cluster run.",
            styles,
        )
    )

    _section("7. Implementation and Methodology", styles, story)
    story.append(
        _paragraph(
            f"On this execution, {preprocessing_summary['rows_before_cleaning']:,} source rows were loaded "
            f"and {preprocessing_summary['rows_after_cleaning']:,} remained after validation and duplicate "
            "removal. The split used "
            f"{preprocessing_summary['train_ratio']:.0%} training data and a fixed seed "
            f"of {preprocessing_summary['random_seed']}. Numerical medians and categorical modes are "
            "estimated from training rows only and applied to both partitions. StringIndexers, OneHotEncoder, "
            "VectorAssembler, StandardScaler, and the classifier are fitted together in each Spark ML Pipeline.",
            styles,
        )
    )
    story.append(
        _paragraph(
            "Logistic Regression estimates a linear decision boundary for the binary target. "
            "Gradient-Boosted Trees combine shallow decision trees sequentially to model nonlinear "
            "feature interactions. Both pipelines are fit only on training data and produce predictions "
            "for the same held-out test partition.",
            styles,
        )
    )
    story.append(
        _paragraph(
            f"Observed source target distribution: {preprocessing_summary['class_distribution']}. "
            "The identifiers used in exported predictions are generated session row IDs, not customer "
            "identities from UCI.",
            styles,
        )
    )

    _section("8. Complete Source Code", styles, story)
    story.append(
        _paragraph(
            "The executable Python sources used for this project are included in the appendices of this PDF. "
            "The workspace files remain the authoritative editable source.",
            styles,
        )
    )

    _section("9. Execution and Output Screenshots", styles, story)
    story.append(
        _paragraph(
            "Authentic screenshots are not generated by the report process. Insert screenshots captured "
            "from your own execution in the marked locations below and save the originals under screenshots/.",
            styles,
        )
    )
    for label in (
        "Project folder structure",
        "Dataset loaded into Spark and schema",
        "Missing-value analysis and preprocessing output",
        "Feature encoding and VectorAssembler",
        "Logistic Regression training",
        "Gradient-Boosted Trees training",
        "Model evaluation metrics",
        "Confusion matrix and ROC curve",
        "Actual-versus-predicted purchases",
        "Dashboard overview and customer segmentation",
        "Interactive session prediction",
        "Successful end-to-end execution",
    ):
        story.extend([_screenshot_placeholder(label), Spacer(1, 5)])

    _section("10. Results and Analysis", styles, story)
    story.append(_metric_table(evaluation))
    story.append(Spacer(1, 8))
    story.append(
        _paragraph(
            f"Selected model: {evaluation['best_model']}. Selection prioritizes purchase-class F1-score, "
            "with ROC-AUC and accuracy as tie-breakers. Accuracy measures the fraction of all correct "
            "classifications; precision measures how many predicted purchases were actual purchases; "
            "recall measures how many actual purchases were detected; F1 balances precision and recall; "
            "ROC-AUC measures ranking quality over decision thresholds. The following values are read "
            "from Spark evaluation on this run's holdout predictions and must not be interpreted as "
            "guarantees of future performance.",
            styles,
        )
    )
    for filename, caption in (
        ("model_comparison.png", "Performance comparison of the two Spark models."),
        ("confusion_matrix.png", "Best-model confusion matrix on the held-out test sessions."),
        ("roc_curve.png", "Best-model ROC curve on the held-out test sessions."),
        ("actual_vs_predicted.png", "Actual and predicted outcome counts for the holdout set."),
    ):
        figure_path = OUTPUTS / filename
        if figure_path.is_file():
            story.extend(
                [
                    Spacer(1, 8),
                    Paragraph(caption, styles["BodyAcademic"]),
                    Image(str(figure_path), width=5.6 * inch, height=3.1 * inch),
                ]
            )
        else:
            story.append(_screenshot_placeholder(caption))

    _section("11. Conclusion", styles, story)
    story.append(
        _paragraph(
            "This project demonstrates a complete local Spark ML workflow using a public, documented "
            "e-commerce session dataset. The trained models and test metrics are generated by execution, "
            "and the Streamlit dashboard reads persisted artifacts rather than retraining on page refresh. "
            "Future work may validate generalization on newer sessions, tune thresholds for business costs, "
            "and use a consented dataset that actually includes demographic and prior-purchase attributes.",
            styles,
        )
    )

    _section("12. Plagiarism Report", styles, story)
    story.append(
        _paragraph(
            "A plagiarism/similarity report must be generated separately using a university-approved "
            "plagiarism-checking tool. No similarity percentage is asserted in this document. Attach the "
            "institutionally generated report here if required.",
            styles,
        )
    )
    story.append(_screenshot_placeholder("University plagiarism-checking report (attach separately)"))

    _section("13. References", styles, story)
    story.append(
        _paragraph(
            "Sakar, C. and Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q.<br/>"
            "Apache Spark documentation: https://spark.apache.org/docs/latest/<br/>"
            "UCI dataset page: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Appendix A - Complete Python Source Code", styles["Heading1"]))
    for source_file in _source_files():
        _appendix_code(source_file, styles, story)

    report = ProjectReportTemplate(destination, styles)
    report.multiBuild(story)
    return destination


if __name__ == "__main__":
    print(generate_report())
