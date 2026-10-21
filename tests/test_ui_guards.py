"""Guards for the dashboard's rules: no math in the UI, one place for charts, and the Streamlit settings.

The rule since Phase 1: ALL math lives in signaldesk/risk.py. The user interface only picks rows and
columns, reshapes tables for the charts, and formats numbers as text. These tests read the SOURCE of
app.py and of EVERY file in the ui package (they do not need the files to be listed: a new file in
ui/ is scanned automatically), and they read the specification of every chart the page draws.
"""

import ast
import re
import subprocess
import sys
import tomllib

import pytest
from app_helpers import APP_PATH, ROOT

UI_DIR = ROOT / "ui"
UI_FILES = [APP_PATH, *sorted(UI_DIR.rglob("*.py"))]  # app.py and every ui file, however many there are

# --- what the scan forbids --------------------------------------------------------------------
# Rule since Phase 1: ALL math lives in signaldesk/risk.py. So the UI's code may not contain
#   - any arithmetic (+ - * / // % ** @, += -= ... , unary minus) on anything that is not a literal,
#     except text/list concatenation and the few lines marked "# not math: layout" or "# not math: date";
#   - any calculating method or function (std, max, round, agg, eval, clip, ...);
#   - a way to hide a calculation (lambda, eval/exec, operator/functools/importlib, dynamic getattr,
#     __import__, a chart transform or aggregate, a Vega expression).
FORBIDDEN_METHODS = {
    # pandas / numpy calculations
    "pct_change", "std", "var", "mean", "median", "cumprod", "cumsum", "cummax", "cummin", "sqrt",
    "quantile", "corr", "cov", "rolling", "expanding", "ewm", "resample", "log", "log1p", "exp",
    "expm1", "prod", "sum", "diff", "shift", "pow", "mul", "div", "sub", "truediv", "floordiv",
    "abs", "rank",
    # more calculations, and the method forms of the operators (df.add(1), df.rsub(1), ...)
    "max", "min", "agg", "aggregate", "eval", "query", "round", "clip", "add", "radd", "rsub", "rmul",
    "rdiv", "rtruediv", "rfloordiv", "rmod", "rpow", "mod", "neg", "nlargest", "nsmallest", "describe",
    "idxmax", "idxmin", "argmax", "argmin", "skew", "kurt", "sem", "corrwith", "accumulate", "reduce",
    "fsum", "hypot",
    # inventing data: the UI never fills a gap or invents a value
    "fillna", "ffill", "bfill", "interpolate",
}  # (.apply is allowed: the Styler uses it to set CSS colours, not to calculate; a lambda is not)
FORBIDDEN_NAMES = {  # built-in functions that calculate, or that run text as code
    "sum", "pow", "abs", "divmod", "max", "min", "round", "eval", "exec", "compile", "__import__", "reduce",
    "globals", "locals", "vars",
}
FORBIDDEN_IMPORTS = {
    "numpy", "math", "statistics", "scipy", "cmath", "decimal", "fractions",  # math libraries
    "operator", "functools", "importlib", "builtins",  # ways to do arithmetic or imports without writing them
}
ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult)
AGGREGATE_SHORTHAND = re.compile(
    r"^\s*(mean|sum|average|median|min|max|count|distinct|stdev|stdevp|variance|variancep|q1|q3)\(", re.I
)
# The window/price-checking functions the charts used to call. Rolling volatility and the correlation
# matrix now come from risk.rolling_volatility and risk.return_correlation, which take prices.
RETIRED_RISK_FUNCTIONS = {"annualized_volatility", "correlation_matrix"}
# The one Vega expression the charts may contain (white text on dark heatmap squares).
ALLOWED_EXPRESSION_TEXT = "abs(datum.Correlation) > 0.6"
# A line (or statement) may be excused with this comment, naming the REASON. Nothing else excuses arithmetic.
NOT_MATH_MARKER = re.compile(r"#\s*not math:\s*(layout|date)\b")

TEXT_METHODS = {"join", "format", "strftime", "upper", "lower", "strip", "title", "removeprefix", "removesuffix", "replace"}


def _is_number(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool)


def _literal_only(node) -> bool:
    """A number, or arithmetic on numbers only (``6 * 60 * 60``, ``-0.05``): it touches no data."""
    if _is_number(node):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _literal_only(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ARITHMETIC):
        return _literal_only(node.left) and _literal_only(node.right)
    return False


def _texty(node) -> bool:
    """Obviously text (or a list/tuple display): a string literal, an f-string, str(...), ", ".join(...), ..."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, bytes))
    if isinstance(node, (ast.JoinedStr, ast.List, ast.Tuple, ast.Set)):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod, ast.Mult)):
        return _texty(node.left) or _texty(node.right)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"str", "repr", "format", "list", "tuple"}:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr in TEXT_METHODS:
            return True
    return False


def _is_concatenation(node) -> bool:
    """``"a" + b``, ``[x] + rest``, ``"%s" % b``, ``"-" * n``, ``path / "name"``: not arithmetic on numbers."""
    if isinstance(node.op, (ast.Add, ast.Mod, ast.Mult)):
        return _texty(node.left) or _texty(node.right)
    if isinstance(node.op, ast.Div):  # a path joined with a name
        return any(isinstance(side, ast.Constant) and isinstance(side.value, str) for side in (node.left, node.right))
    return False


def marked_lines(source: str) -> dict[int, str]:
    """Line number -> reason, for every line that carries a "# not math: layout|date" comment."""
    marked = {}
    for number, line in enumerate(source.splitlines(), start=1):
        found = NOT_MATH_MARKER.search(line)
        if found:
            marked[number] = found.group(1)
    return marked


def find_math(source: str) -> list[str]:
    """Look at the CODE of a Python source (not its comments) for calculations. Returns the findings."""
    tree = ast.parse(source)
    marked = marked_lines(source)

    def excused(node) -> bool:
        return any(number in marked for number in range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1))

    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_METHODS:
                found.append(f"line {node.lineno}: .{node.attr}")
            if node.attr.startswith("transform"):  # Altair: chart.transform_calculate(...), ...
                found.append(f"line {node.lineno}: .{node.attr} (a chart transform)")
            if node.attr in RETIRED_RISK_FUNCTIONS:
                found.append(f"line {node.lineno}: .{node.attr} (use the price-taking risk.py function)")
            if node.attr == "today":  # date.today() / Timestamp.today(): the page uses data.market_today()
                found.append(f"line {node.lineno}: .today (use data.market_today())")
            if node.attr in {"now", "utcnow"}:
                found.append(f"line {node.lineno}: .{node.attr} (use data.market_today())")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in FORBIDDEN_NAMES:
            found.append(f"line {node.lineno}: {node.id}(...)")
        if isinstance(node, ast.Lambda):
            found.append(f"line {node.lineno}: lambda (hides a calculation: write a named function, and no math)")
        if isinstance(node, ast.keyword) and node.arg in {"aggregate", "calculate", "transform"}:
            found.append(f"line {node.value.lineno}: {node.arg}= (chart-side math)")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ARITHMETIC):
            if not (_literal_only(node) or _is_concatenation(node) or excused(node)):
                found.append(f"line {node.lineno}: arithmetic ({type(node.op).__name__})")
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ARITHMETIC):
            text_append = isinstance(node.op, ast.Add) and _texty(node.value)
            if not (text_append or excused(node)):
                found.append(f"line {node.lineno}: augmented assignment ({type(node.op).__name__})")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            if not (_literal_only(node) or excused(node)):
                found.append(f"line {node.lineno}: unary sign change")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"getattr", "setattr", "delattr"}:
            name = node.args[1] if len(node.args) > 1 else None
            if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
                found.append(f"line {node.lineno}: {node.func.id} with a name that is not a plain string")
            elif name.value in FORBIDDEN_METHODS or name.value.startswith("transform"):
                found.append(f"line {node.lineno}: {node.func.id}(..., {name.value!r}) reaches a forbidden method")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if AGGREGATE_SHORTHAND.match(node.value):
                found.append(f"line {node.lineno}: {node.value!r} (Altair aggregate shorthand)")
            if "datum." in node.value and node.value != ALLOWED_EXPRESSION_TEXT:
                found.append(f"line {node.lineno}: {node.value!r} (a chart expression)")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    found.append(f"line {node.lineno}: import {alias.name} (a math or code-running library)")
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                found.append(f"line {node.lineno}: from {node.module} import ... (a math or code-running library)")
            for alias in node.names:
                if alias.name in FORBIDDEN_METHODS or alias.name in FORBIDDEN_NAMES:
                    found.append(f"line {node.lineno}: from ... import {alias.name} (a calculating function)")
    return found


def test_the_scan_covers_app_py_and_every_ui_file():
    names = {path.name for path in UI_FILES}
    assert {"app.py", "__init__.py", "analysis.py", "charts.py", "formatting.py", "inputs.py", "live.py",
            "sections.py", "state.py"} <= names
    assert all(path.is_file() for path in UI_FILES)
    assert len(UI_FILES) >= 9


@pytest.mark.parametrize("path", UI_FILES, ids=lambda path: path.name)
def test_ui_files_contain_no_financial_math(path):
    """All math lives in signaldesk/risk.py, so the UI's code must not calculate anything."""
    found = find_math(path.read_text())
    assert not found, f"{path.name} must not calculate numbers itself: {found}"


# Every snippet type here once slipped through (or could): the scan must flag each one.
SNEAKY_SNIPPETS = {
    # arithmetic on data, in every form
    "a - b": "d = a - b",
    "a + b": "d = a + b",
    "a * b": "d = a * b",
    "a // b": "d = a // b",
    "a % b": "d = a % b",
    "a ** b": "d = a ** b",
    "a @ b": "d = a @ b",
    "unary minus": "d = -x",
    "unary plus": "d = +x",
    "negated call": "d = -df['x']",
    "vol / 252": "d = vol / 252",
    "x * 1.1": "d = x * 1.1",
    "x * 100": "f = df * 100",
    "x / 100": "f = weight / 100",
    "100.0 * x": "f = 100.0 * weight",
    "x += 1": "x += 1",
    "x -= y": "x -= y",
    "x *= 100": "x *= 100",
    "x /= 100": "x /= 100",
    "x **= 2": "x **= 2",
    "arithmetic inside an f-string": "t = f'{a - b:.1%}'",
    "arithmetic inside a call": "t = show(a - b)",
    "arithmetic in a comprehension": "t = [a - 1 for a in values]",
    "arithmetic with an unrelated marker": "d = a - b  # not math: because I say so",
    # calculating methods and functions
    ".max()": "y = df.max()",
    ".min()": "y = df.min()",
    ".agg('std')": "y = df.agg('std')",
    ".aggregate": "y = df.aggregate('std')",
    ".add()": "y = df.add(1)",
    ".rsub()": "y = df.rsub(1)",
    ".round()": "y = df.round(2)",
    ".clip()": "y = df.clip(0, 1)",
    ".nlargest": "y = df.nlargest(3, 'x')",
    ".nsmallest": "y = df.nsmallest(3, 'x')",
    ".describe()": "y = df.describe()",
    "df.eval": "y = df.eval('a - b')",
    "pd.eval": "y = pd.eval('a - b')",
    ".query with an expression": "y = df.query('a - b > 0')",
    ".fillna (invents data)": "y = df.fillna(0)",
    ".ffill (invents data)": "y = df.ffill()",
    ".interpolate (invents data)": "y = df.interpolate()",
    "pct_change": "x = df.pct_change()",
    ".cummin": "y = df.cummin()",
    ".sum": "y = df.sum()",
    ".mean": "y = df.mean(axis=1)",
    ".diff": "y = df.diff()",
    ".mul": "z = df.mul(100)",
    ".resample": "z = df.resample('W')",
    ".pow": "z = df.pow(2)",
    ".rolling": "z = df.rolling(20).std()",
    "max()": "z = max(a, b)",
    "min()": "z = min(a)",
    "round()": "z = round(x, 2)",
    "sum()": "g = sum(weights)",
    "abs()": "g = abs(x)",
    "pow()": "g = pow(2, 3)",
    "divmod()": "g = divmod(seconds, 60)",
    # ways to hide a calculation
    "lambda in .apply": "y = df.apply(lambda c: c)",
    "lambda anywhere": "f = lambda x: x",
    "operator.add": "import operator\ny = operator.add(a, b)",
    "operator.neg": "import operator\ny = operator.neg(a)",
    "from operator import": "from operator import add",
    "functools.reduce": "import functools\ny = functools.reduce(f, xs)",
    "from functools import reduce": "from functools import reduce",
    "reduce()": "y = reduce(f, xs)",
    "itertools.accumulate": "import itertools\ny = list(itertools.accumulate(xs))",
    "from itertools import accumulate": "from itertools import accumulate",
    "getattr with a built name": "y = getattr(c, 'transform_' + 'calculate')",
    "getattr with a variable": "y = getattr(c, name)",
    "getattr of a forbidden method": "y = getattr(df, 'std')",
    "setattr with a variable": "setattr(c, name, v)",
    "__import__": "np = __import__('numpy')",
    "importlib": "import importlib\nnp = importlib.import_module('numpy')",
    "from importlib import": "from importlib import import_module",
    "eval()": "y = eval('1 + 1')",
    "exec()": "exec('y = 1')",
    "compile()": "y = compile('1', 'x', 'eval')",
    # math libraries
    "import numpy": "import numpy as np",
    "from math": "from math import sqrt",
    "import statistics": "import statistics",
    "import decimal": "import decimal",
    # dates and old routes
    "date.today()": "x = date.today()",
    "Timestamp.today()": "x = pd.Timestamp.today()",
    "datetime.now()": "x = datetime.now()",
    "old window route": "v = risk.annualized_volatility(returns, window=60)",
    "old correlation route": "c = risk.correlation_matrix(returns)",
    # chart-side math
    "transform_calculate": "c = chart.transform_calculate(a='datum.a * 100')",
    "transform_window": "c = chart.transform_window(a='sum(a)')",
    "aggregate shorthand": "d = alt.Y('mean(Value):Q')",
    "aggregate keyword": "e = alt.Y(field='v', aggregate='mean')",
    "a Vega expression": "c = alt.condition('datum.Value > 1', alt.value('red'), alt.value('blue'))",
}

# Ordinary code the scan must leave alone.
HARMLESS_SNIPPETS = {
    "text concatenation": "t = 'a' + name",
    "text concatenation, text on the right": "t = name + 'a'",
    "concatenation with str()": "t = prefix + str(n)",
    "an f-string": "t = f'{a:.1%} of {b}'",
    "list concatenation": "t = [a] + rest",
    "tuple concatenation": "t = (a,) + rest",
    "text formatting": "t = '%s items' % n",
    "text repetition": "t = '-' * 20",
    "a path joined with a name": "p = root / 'data'",
    "appending text": "caption += ' more words'",
    "appending an f-string": "caption += f' {n} more'",
    "a negative literal": "x = -1\ny = -0.05",
    "arithmetic on literals only": "SECONDS = 6 * 60 * 60",
    "boolean not and invert": "k = ~mask\nj = not flag",
    "a marked layout line": "height = 64 * len(tickers)  # not math: layout",
    "a marked date line": "end_exclusive = end + timedelta(days=1)  # not math: date",
    "a marked multi-line statement": "x = foo(\n    a - b,  # not math: date\n)",
    "apply with a named function": "styled = table.apply(grey_out, axis=1)",
    "getattr with a plain name": "y = getattr(obj, 'label')",
    "the market date helper": "d = data.market_today()",
    "the price-taking risk functions": "v = risk.rolling_volatility(prices, 60)\nc = risk.return_correlation(prices)",
    "comparisons": "ok = a < b <= c\nsame = a == b",
    "the one allowed chart expression": "c = alt.condition('abs(datum.Correlation) > 0.6', alt.value('white'), alt.value('black'))",
    "a Timedelta rounded up by pandas": "w = pd.Timedelta(seconds=s).ceil('min')",
}


@pytest.mark.parametrize("name", list(SNEAKY_SNIPPETS))
def test_the_math_scan_catches_every_sneaky_snippet(name):
    assert find_math(SNEAKY_SNIPPETS[name]), f"the scan missed: {name}"


@pytest.mark.parametrize("name", list(HARMLESS_SNIPPETS))
def test_the_math_scan_leaves_ordinary_code_alone(name):
    assert find_math(HARMLESS_SNIPPETS[name]) == [], name


def test_the_not_math_marker_excuses_only_its_own_lines_and_only_for_a_stated_reason():
    code = "a = 1 * b  # not math: layout\nc = d - e\n"  # the marker covers line 1, not line 2
    findings = find_math(code)
    assert len(findings) == 1 and findings[0].startswith("line 2:")
    assert find_math("a = b - c  # not math: because\n")  # an unknown reason excuses nothing
    assert find_math("a = b - c  # not math\n")
    assert find_math("a = b - c  # this is fine\n")
    assert marked_lines("a = 1\nb = 2 * c  # not math: layout\nd = e + f  # not math: date\n") == {2: "layout", 3: "date"}


def test_the_two_planted_bugs_the_reviewer_slipped_past_the_old_guard_are_caught():
    # M31: a new file computing "Ahead of SPY by X% a year" from two numbers.
    extra = (
        "def ahead(portfolio, benchmark):\n"
        "    return portfolio - benchmark\n"
        "\n"
        "def label(portfolio, benchmark):\n"
        "    return f'Ahead of SPY by {ahead(portfolio, benchmark):.1%} a year'\n"
    )
    assert find_math(extra) == ["line 2: arithmetic (Sub)"]
    # M32: format_percent multiplying in place.
    percent = (
        "def format_percent(value):\n"
        "    value *= 100\n"
        "    return f'{value:,.1f}%'\n"
    )
    assert find_math(percent) == ["line 2: augmented assignment (Mult)"]


def test_the_real_ui_excuses_only_a_few_known_lines():
    """Adding a new "# not math" line is a decision to review, so it needs a change in this test."""
    excused = {}
    for path in UI_FILES:
        for reason in marked_lines(path.read_text()).values():
            excused[(path.name, reason)] = excused.get((path.name, reason), 0) + 1
    assert excused == {
        ("charts.py", "layout"): 1,  # the heatmap's height per ticker
        ("inputs.py", "date"): 1,  # the picker's end date is included: load_prices wants the day after
        ("live.py", "date"): 2,  # how old a memory entry is, in seconds
    }


def charts_made_in(path) -> set[str]:
    """Names of the functions in a file that build an Altair chart (they call alt.Chart or alt.layer)."""
    tree = ast.parse(path.read_text())
    builders = set()
    for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "alt"
                and node.func.attr in {"Chart", "layer", "hconcat", "vconcat", "concat"}
            ):
                builders.add(function.name)
    return builders


def test_charts_are_built_in_one_file_and_every_builder_is_scanned_by_the_tests():
    """Altair appears only in ui/charts.py, and its builders are the ones test_ui_units.py scans for chart-side math.

    If someone adds a chart builder (here or anywhere else in the UI), this fails until the new builder
    is added to the chart-spec scan too.
    """
    for path in UI_FILES:
        text = path.read_text()
        uses_altair = bool(re.search(r"^\s*(import altair|from altair)", text, re.M))
        if path.name != "charts.py":
            assert not uses_altair, f"{path.name} imports altair: build charts in ui/charts.py, where they are scanned"
    assert charts_made_in(UI_DIR / "charts.py") == {"line_chart", "correlation_chart"}
    for path in UI_FILES:
        if path.name != "charts.py":
            assert charts_made_in(path) == set(), path.name


def test_every_chart_the_page_draws_is_scanned_for_chart_side_math(monkeypatch, app_env):
    """Draw the page in its different shapes and scan the specification of every chart that comes out."""
    from app_helpers import (
        assert_no_chart_math,
        chart_spec,
        charts,
        install_demo,
        make_demo,
        run_app,
    )

    install_demo(monkeypatch, make_demo())
    at = run_app()
    drawn = list(charts(at))
    at.sidebar.toggle(key="show_holdings").set_value(True).run()
    drawn += list(charts(at))
    at.sidebar.slider(key="window").set_value(30).run()
    drawn += list(charts(at))

    assert len(drawn) == 12  # four charts, three times
    for chart in drawn:
        assert_no_chart_math(chart_spec(chart))


# --------------------------------------------------------------------------- #
# .streamlit/config.toml
# --------------------------------------------------------------------------- #

CONFIG_PATH = ROOT / ".streamlit" / "config.toml"


def test_streamlit_config_exists_with_the_public_site_settings():
    assert CONFIG_PATH.is_file()
    config = tomllib.loads(CONFIG_PATH.read_text())

    assert config["client"]["showErrorDetails"] == "none"  # a visitor never sees a traceback
    assert config["client"]["toolbarMode"] == "viewer"  # no developer menu for visitors
    assert config["browser"]["gatherUsageStats"] is False
    theme = config["theme"]
    assert theme["base"] == "light"
    for key in ("primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor"):
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", theme[key]), key
    assert theme["backgroundColor"].lower() == "#ffffff"  # a light theme: white page
    assert theme["primaryColor"].lower() != theme["backgroundColor"].lower()


def test_streamlit_itself_accepts_the_config_file():
    """`streamlit config show` reads the file in the project folder; a misspelled key would not show up."""
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "config", "show"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    effective = {line.strip() for line in result.stdout.splitlines() if not line.lstrip().startswith("#")}
    for line in (
        'showErrorDetails = "none"',
        'toolbarMode = "viewer"',
        "gatherUsageStats = false",
        'base = "light"',
    ):
        assert line in effective, line
    assert "not a valid" not in (result.stdout + result.stderr).lower()
