"""Build Online Resource 1 from the recorded campaign, without running a solver."""
import csv
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT


def number_or_text(value):
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def main():
    destination = ROOT / "Submission" / "ESM_1.xlsx"
    book = Workbook()
    cover = book.active
    cover.title = "Guide"
    for row in [
        ("Online Resource 1", "Fewer Task Errors, More Systemic Failures in AI Agent Workflows"),
        ("Journal", "Autonomous Agents and Multi-Agent Systems"),
        ("Authors", "Muting Ma and Jiaqi Gong"),
        ("Correspondence", "Muting Ma, The University of Alabama, Tuscaloosa, AL, USA. mma10@ua.edu"),
        ("X1 to X7", "Plan means by experimental setting. n is the replication count."),
        ("Test samples", "50,000 scenarios per job in X1 to X6 and 20,000 for EF and BD in X7."),
        ("Units", "Probabilities and shares are fractions. Solver times are seconds."),
        ("Loss", "exp_loss is weighted loss. wrong_per_run counts all released wrong outputs."),
        ("Risk", "cvar is the worst-decile mean loss. risk_obj averages exp_loss and cvar."),
        ("Missing values", "A blank cell means the statistic is inapplicable or was not recorded."),
        ("Paired tests", "Within-workflow differences, pointwise 95% intervals, and Holm-adjusted p values."),
        ("Model sizes", "Variables and constraints in each of the 60 extensive forms built in X7."),
        ("Perfect information", "Minimum training loss for nine X7 jobs with scenario-specific reviews. Status 2 means optimal."),
        ("Protocol checks", "Planning comparisons specified before the X1 to X6 experiments and their observed outcomes."),
    ]:
        cover.append(row)
    for name, filename in [(f"X{i}", f"X{i}_summary.csv") for i in range(1, 8)] + [
        ("Paired tests", "paired.csv"), ("Protocol checks", "expectations.csv")
    ]:
        sheet = book.create_sheet(name)
        with (RUN / "tables" / filename).open(newline="") as source:
            reader = csv.DictReader(source)
            columns = [field for field in reader.fieldnames if field not in {"solver_gap", "share_gap_le_5pct"}]
            sheet.append(columns)
            for row in reader:
                if name == "X7" and row["cell"].endswith("_PI"):
                    continue
                if name == "Protocol checks" and row["block"] == "X7":
                    continue
                sheet.append([number_or_text(row[field]) for field in columns])
    sizes = book.create_sheet("Model sizes")
    sizes.append(["Replication", "Tasks", "Training scenarios", "Variables", "Constraints"])
    for path in sorted((RUN / "runs" / "X7").glob("rep_*/*_EF.json")):
        job = json.loads(path.read_text())
        solver = job["result"]["plans"]["EF"]
        sizes.append([job["rep"], job["result"]["n"], job["S"], solver["solver_cols"], solver["solver_rows"]])
    pi = book.create_sheet("Perfect information")
    pi.append(["Replication", "Tasks", "Training scenarios", "Training objective", "Status"])
    for path in sorted((RUN / "runs" / "X7").glob("rep_*/*_PI.json")):
        job = json.loads(path.read_text())
        result = job["result"]["pi"]
        pi.append([job["rep"], job["result"]["n"], job["S"], result["obj"], result["status"]])
    for sheet in book:
        sheet.freeze_panes = "A2"
        if sheet != cover:
            sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="17364C")
        for column in sheet.columns:
            width = max(len(str(cell.value or "")) for cell in column)
            sheet.column_dimensions[get_column_letter(column[0].column)].width = min(width + 3, 110)
    destination.parent.mkdir(parents=True, exist_ok=True)
    book.save(destination)


if __name__ == "__main__":
    main()
