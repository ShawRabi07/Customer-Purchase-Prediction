# Predicting Customer Purchase Behaviour Using Apache Spark and Machine Learning

INT315 - Cluster Computing (CA2)

This project trains and evaluates Spark MLlib classifiers for predicting whether
an online-shopping session ends in a purchase. It downloads the real UCI Online
Shoppers Purchasing Intention Dataset, saves the fitted preprocessing/model
pipelines, exports holdout predictions and metrics, provides a Streamlit dashboard,
and generates a PDF report from actual execution artifacts.

## Dataset and scope

- **Dataset:** [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Direct download:** [UCI ZIP archive](https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip)
- **Citation:** Sakar, C. and Kastro, Y. (2018). UCI Machine Learning Repository. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).
- The dataset contains 12,330 web sessions and the `Revenue` outcome. The loader maps `Revenue=True` to `purchase=1` and `False` to `purchase=0`.
- It contains browsing/page counts, category-specific duration, web analytics, month, technical/traffic categories, visitor type, and weekend status.
- **Important limitations:** It does not contain customer age, age group, prior purchase count, or a separately measured session-duration field. Session duration in this project is derived from the three available page-category duration columns. `VisitorType` is only a new/returning visitor category, not a purchase-history field. These unavailable values are not invented.
- The exported `customer_id` is an anonymous generated session row identifier because UCI does not provide a customer ID.
- The dataset is licensed CC BY 4.0; retain the attribution above.

The loader downloads the official archive on the first run and reuses the validated
CSV on later runs. Use `--refresh-data` to download it again.

## Technology

Python 3.10+, Java 17+, Apache Spark/PySpark 3.5 or 4.x in **local mode**,
Pandas, NumPy, scikit-learn (ROC curve coordinates), Matplotlib, Seaborn,
Streamlit, Plotly, and ReportLab. The workflow uses `local[*]`; it is not a
multi-node cluster execution. The Spark DataFrame and MLlib pipeline can be
adapted to a cluster by changing the Spark deployment configuration.

## Project structure

```text
customer-purchase-prediction/
├── data/customer_behaviour.csv
├── dashboard/app.py
├── models/                         # Saved Spark PipelineModels after training
├── notebooks/                      # Optional EDA workspace
├── outputs/
│   ├── model_metrics.json
│   ├── preprocessing_summary.json
│   ├── purchase_predictions.csv
│   └── diagnostic PNG files
├── report/
│   ├── generate_report.py
│   └── project_report.pdf          # Generated after a successful run
├── screenshots/CHECKLIST.md
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── prediction.py
│   └── spark_session.py
├── tests/test_data_loader.py
├── main.py
└── requirements.txt
```

## Windows setup (PowerShell)

1. Install a 64-bit JDK 17 distribution and ensure `java -version` works.
2. Install `winutils.exe` matching the Hadoop version bundled with PySpark from
   an approved source. Set `JAVA_HOME` and, for native Windows, `HADOOP_HOME` to
   the folder containing `bin\winutils.exe`:

   ```powershell
   $env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17"
   $env:Path = "$env:JAVA_HOME\bin;$env:Path"
   $env:HADOOP_HOME = "C:\Winutils"
   $env:Path = "$env:HADOOP_HOME\bin;$env:Path"
   java -version
   ```

   The application can infer `HADOOP_HOME` when `winutils.exe` is already on
   `PATH`; setting it explicitly is recommended. Spark uses Hadoop filesystem
   operations when saving fitted models, so native Windows runs need this binary.
   The exact Hadoop version is bundled in your installed PySpark package and can
   change when PySpark is upgraded.

3. Create and activate a virtual environment, then install dependencies:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` and
   `.\.venv\Scripts\streamlit.exe` directly.

PySpark provides the Spark runtime through its Python package; a separate
multi-node Spark installation is not needed for this local workflow. On Windows,
use a supported JDK and the matching Hadoop Windows binary for filesystem operations.

### Troubleshooting `JAVA_GATEWAY_EXITED`

This error means PySpark could not start its Java process. In the same PowerShell
window you will use to run the application, point `JAVA_HOME` at the JDK
installation directory (not its `bin` folder), and put that JDK first on `PATH`:

```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
java -version
Test-Path "$env:JAVA_HOME\bin\java.exe"
```

Use the actual JDK path installed on your computer. Then verify Spark starts
before launching the dashboard:

```powershell
.\.venv\Scripts\python.exe -c "from src.spark_session import create_spark_session; s=create_spark_session(); print(s.version); s.stop()"
```

If this command fails, check that the JDK is installed, `JAVA_HOME` points to its
root directory, and the project's virtual environment has the requirements
installed. Start Streamlit with that same environment so it uses the same
Python/PySpark installation:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

## Run the workflow

From the project root:

```powershell
python main.py
```

The first run downloads and validates the UCI ZIP into `data\customer_behaviour.csv`.
To force a fresh download:

```powershell
python main.py --refresh-data
```

To select another local CSV path while still using the expected UCI schema:

```powershell
python main.py --dataset-path data\customer_behaviour.csv
```

The main workflow prints the Spark schema and sample, reports null counts and
class distribution, cleans and splits the records, fits both models, evaluates
the holdout predictions, saves the best-model predictions, and generates
`report\project_report.pdf`.

Run the dashboard after a successful training run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The dashboard loads the saved dataset, metrics, predictions, and model pipeline.
It does not retrain models when the page is refreshed. Its one-session form uses
the exact saved Spark transformations; inputs correspond only to actual UCI fields.

Run focused tests:

```powershell
python -m unittest discover -s tests -v
```

## Processing and model workflow

1. Download the official archive and validate its columns and binary target.
2. Read the CSV into Spark; cast schema, remove duplicate records, discard invalid
   negative counts/durations and out-of-range rate fields, and retain valid labels.
3. Split into 80% training and 20% holdout using seed 42. Fit numeric median and
   categorical mode imputation using training data only, then apply those values
   to both partitions.
4. Fit `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, and `StandardScaler`
   as stages in each model pipeline. Pipeline estimators are fitted on training
   data only.
5. Train Spark MLlib Logistic Regression and Gradient-Boosted Trees.
6. Evaluate holdout data with Spark accuracy, purchase-class precision, recall,
   F1, and ROC-AUC evaluators. Select the model by purchase-class F1, then
   ROC-AUC and accuracy as tie-breakers.
7. Save both complete Spark pipelines and export predictions and diagnostic charts.

Precision answers how many predicted purchases were actual purchases; recall
answers how many actual purchases were detected; F1 balances the two; accuracy
is the share of all classifications correct; ROC-AUC summarizes ranking quality
across decision thresholds. The run's actual values are written to
`outputs\model_metrics.json`; results are not pre-filled or fabricated.

## Dashboard

- Interactive month, visitor type, visit day, and actual-outcome filters update session
  KPIs, purchase analysis, segmentation, and matching holdout predictions.
- Purchase/no-purchase distribution and conversion by browsing activity, derived
  session duration, visitor type, or month.
- Model metric comparison, confusion matrix, ROC curve, and actual/predicted charts.
- Searchable holdout predictions with probability and correctness filters; download
  the currently filtered results as CSV.
- Optional single-session inference through the saved best Spark pipeline.

The source does not include age or previous purchase count, so age-group and
purchase-history charts and inputs are deliberately omitted. The dashboard labels
`VisitorType` as a limited proxy and never equates it with purchase history.

## Generated files and report

- `models\logistic_regression\` and `models\gradientboosted_trees\`: full saved pipelines.
- `outputs\purchase_predictions.csv`: generated session ID, observed features,
  actual/predicted purchase, probability, and correctness.
- `outputs\model_metrics.json` and `outputs\preprocessing_summary.json`: actual
  run measurements.
- `outputs\confusion_matrix.png`, `roc_curve.png`, `model_comparison.png`, and
  `actual_vs_predicted.png`.
- `report\project_report.pdf`: report with real run metrics, code appendices, and
  clearly marked screenshot placeholders.
- `screenshots\CHECKLIST.md`: screenshot capture checklist.

No execution screenshots or plagiarism scores are fabricated. Generate any
plagiarism/similarity report separately using a university-approved service.

## Screenshots

Save genuine screenshots in `screenshots\` and follow
[the screenshot checklist](screenshots/CHECKLIST.md). The PDF contains placeholders
for screenshots that must be captured by the student.

## Future improvements

- Evaluate temporal or external validation to measure deployment generalization.
- Tune decision thresholds and model hyperparameters against business costs.
- Use appropriately consented data with explicit demographics and purchase history
  if age-based or prior-purchase analysis is required.
- Deploy Spark on a multi-node cluster and compare with local-mode performance.
