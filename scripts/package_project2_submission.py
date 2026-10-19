"""Package two independent Project 2 studies from an already committed HEAD.

No data/model API, original vendor data, or Blackboard submission is involved.
Validation completes before the destination is created. Existing files are never
overwritten or deleted; a failed write leaves its partial output for inspection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlsplit
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MEMBERS = (("12412408", "胡博文"), ("12410819", "谢明飞"), ("12410436", "王天希"))
REPOSITORY_URL = "https://github.com/caolvchong218/cf2026-quant-platform"
STUDIES = {
    "RSRS": {
        "key": "project2_rsrs", "prefix": "CF2026_Project2",
        "audit": "PROJECT2_SOURCE_AUDIT.md", "script": "PROJECT2_PRESENTATION_SCRIPT.md",
    },
    "CogAlpha": {
        "key": "project2_cogalpha", "prefix": "CF2026_Project2_CogAlpha",
        "audit": "PROJECT2_PAPER2_SOURCE_AUDIT.md", "script": "PROJECT2_COGALPHA_PRESENTATION_SCRIPT.md",
    },
}
MATERIALS = ("Report.pdf", "Beamer.pdf", "Presentation.pptx", "Speaker_Script.pdf")
TEX_FILES = ("final_report.tex", "beamer.tex", "speaker_script.tex")
COMMON_DOCS = ("PROJECT2_REQUIREMENTS_MATRIX.md", "PROJECT2_REPRODUCIBILITY.md")
ERRATUM = "configs/project2_cogalpha_source_erratum.json"
AUDIT_SCRIPT = "scripts/audit_project2_cogalpha.py"
FINAL_AUDIT = "evidence/project2_release/cogalpha_release_audit.json"
FINAL_UI = "evidence/project2_release/cogalpha_ui_acceptance.json"
INTERMEDIATE_SUFFIXES = {".aux", ".log", ".out", ".toc", ".nav", ".snm", ".fls", ".fdb_latexmk"}
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".toml", ".yaml", ".yml", ".md", ".txt", ".ini", ".cfg", ".cmd", ".ps1", ".sh"}
FITTED_MODEL_FIELDS = {"coefficients", "intercept", "coef_", "intercept_", "state_dict", "model_weights",
                       "weights", "tree_info", "tree_structure", "leaf_value", "model_dump", "booster_model"}


class PackageError(RuntimeError):
    pass


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def git(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", *arguments],
            cwd=root, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        # Do not echo external command output that could contain remote credentials.
        raise PackageError(f"Git operation failed ({arguments[0]}), exit {exc.returncode}.") from None


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise PackageError("Unsafe relative file path in archive or manifest.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise PackageError("Unsafe relative file path in archive or manifest.")
    return path.as_posix()


def public_path(path: str) -> bool:
    """Reject private runtime paths and likely credential files, not API source code."""
    parts = PurePosixPath(path).parts
    lower = [part.lower() for part in parts]
    leaf = lower[-1]
    if any(part in {"runs", "tmp", "private", "env", "venv", "__pycache__", ".git", ".aws", ".ssh", ".codex", "secrets", "credentials"}
           or part.startswith(".venv") or part == ".env" or part.startswith(".env.") for part in lower):
        return False
    if any(lower[i:i + 2] in (["data", "raw"], ["data", "processed"]) for i in range(len(lower) - 1)):
        return False
    if leaf == ".env" or leaf.startswith(".env.") or leaf in {"id_rsa", "id_ed25519", "credentials", ".netrc", "_netrc"}:
        return False
    if Path(leaf).suffix in {".pem", ".key", ".p12", ".pfx", ".parquet", ".pkl", ".pickle", ".joblib", ".onnx", ".pt", ".pth"}:
        return False
    if Path(leaf).suffix in {".txt", ".json", ".yaml", ".yml", ".ini", ".cfg"} and re.search(
            r"(^|[_.-])(token|secret|credentials?|password|api[_-]?key)([_.-]|$)", leaf):
        return False
    # Source reports are linked, not redistributed wholesale. Generated reports
    # and figure PDFs remain allowed under the reports tree.
    if leaf.endswith(".pdf") and (not lower[0] == "reports" or any(
            part in {"sources", "source", "reference", "references"} for part in lower)
            or "everbright" in leaf or "2511.18850" in leaf):
        return False
    return True


def credential_findings(path: str, content: bytes) -> list[str]:
    """Conservative literal scan; return paths/reasons only, never matched values."""
    if PurePosixPath(path).suffix.lower() not in TEXT_SUFFIXES:
        return []
    text = content.decode("utf-8", errors="replace")
    findings = []
    if re.search(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", text):
        findings.append("private-key block")
    if re.search(r"\bsk-[A-Za-z0-9]{32,}\b", text):
        findings.append("possible live API credential")
    pattern = r'''(?i)(?:api[_-]?key|access_token|tushare_token|deepseek_api_key|openai_api_key|password)["']?\s*[:=]\s*["']([A-Za-z0-9+/=_-]{24,})["']'''
    for match in re.finditer(pattern, text):
        value = match.group(1).lower()
        if not any(marker in value for marker in ("fixture", "example", "dummy", "placeholder", "changeme", "your_api", "your_token", "for_testing")):
            findings.append("possible embedded credential literal")
            break
    return findings


def learned_model_findings(path: str, content: bytes) -> list[str]:
    """Public CogAlpha evidence may describe a fit but must not publish its weights."""
    if not path.startswith("evidence/project2_cogalpha/"):
        return []
    leaf = PurePosixPath(path).name.lower()
    if re.search(r"(^|[_.-])(coefficients|intercept|model_weights|model_dump|state_dict)([_.-]|$)", leaf):
        return ["fitted-model payload filename"]
    if leaf.endswith(".json"):
        value = json.loads(content)
        findings = set()

        def inspect(item):
            if isinstance(item, dict):
                for key, child in item.items():
                    if str(key).lower() in FITTED_MODEL_FIELDS:
                        findings.add(str(key))
                    inspect(child)
            elif isinstance(item, list):
                for child in item:
                    inspect(child)

        inspect(value)
        return ["fitted-model field: " + field for field in sorted(findings)]
    if leaf.endswith(".txt"):
        text = content.decode("utf-8", errors="replace")
        if re.search(r"(?m)^Tree=\d+\s*$", text) and "leaf_value=" in text:
            return ["serialized LightGBM tree payload"]
    return []


def validate_source_erratum(root: Path) -> dict:
    """Keep the frozen protocol identity while checking the independent correction."""
    config_path = regular_file(root, "configs/project2_cogalpha.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    erratum_path = regular_file(root, ERRATUM)
    erratum = json.loads(erratum_path.read_text(encoding="utf-8"))
    reference = json.loads(regular_file(root, "evidence/project2_cogalpha/source_reference.json").read_text(encoding="utf-8"))
    corrected = erratum.get("correct_pdf_sha256")
    if not isinstance(corrected, str) or not re.fullmatch(r"[a-f0-9]{64}", corrected):
        raise PackageError("Source erratum must provide the corrected 64-character PDF SHA-256.")
    if (erratum.get("frozen_config") != "configs/project2_cogalpha.json"
            or erratum.get("frozen_config_sha256") != sha256(config_path.read_bytes())
            or erratum.get("incorrect_frozen_value") != config.get("paper_sha256")
            or corrected == config.get("paper_sha256")):
        raise PackageError("Source erratum does not match the frozen study configuration.")
    if reference.get("pdf_sha256") != corrected or reference.get("source_identity_erratum") != ERRATUM:
        raise PackageError("Public paper source reference does not use the independent erratum.")
    manifest = json.loads(regular_file(root, "evidence/project2_cogalpha/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_metadata_erratum_sha256") != sha256(erratum_path.read_bytes()):
        raise PackageError("Final CogAlpha manifest does not seal the source identity erratum.")
    return {"file": ERRATUM, "sha256": sha256(erratum_path.read_bytes()),
            "frozen_config_preserved": True, "corrected_pdf_sha256": corrected}


def validate_release_audit(root: Path) -> dict:
    """Require the completed stored-evidence audit, not its synthetic self-test."""
    path = regular_file(root, FINAL_AUDIT)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("schema") != "qingxu.project2.cogalpha.release_audit.v1"
            or report.get("passed") is not True or report.get("audit_executed") is not True
            or report.get("status") != "passed"):
        raise PackageError("CogAlpha formal release audit has not passed.")
    if report.get("audit_script_sha256") != sha256(regular_file(root, AUDIT_SCRIPT).read_bytes()):
        raise PackageError("Formal audit receipt belongs to a different audit script.")
    required_sections = {"identities_and_release_hashes", "proposal_slots_and_failures",
                         "training_selection_and_models", "api_budget_and_receipts",
                         "ninety_stored_portfolios", "eighteen_comparisons"}
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks or not all(isinstance(check, dict) and check.get("passed") is True for check in checks):
        raise PackageError("Formal audit contains incomplete or failed checks.")
    if not required_sections.issubset({check.get("section") for check in checks}):
        raise PackageError("Formal audit is missing required study sections.")
    manifest = json.loads(regular_file(root, "evidence/project2_cogalpha/manifest.json").read_text(encoding="utf-8"))
    identity = report.get("details", {}).get("identities_and_release_hashes", {}).get("study_identity")
    if identity != manifest.get("identity") or not identity:
        raise PackageError("Formal audit and CogAlpha evidence have different study identities.")
    return {"file": FINAL_AUDIT, "sha256": sha256(path.read_bytes()), "passed": True,
            "audit_script_sha256": report["audit_script_sha256"], "study_identity": identity}


def validate_ui_acceptance(root: Path) -> dict:
    """Bind the actual browser acceptance to the released UI and evidence."""
    path = regular_file(root, FINAL_UI)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("schema") != "qingxu.project2.cogalpha_ui_acceptance.v1"
            or report.get("passed") is not True
            or report.get("screenshots_visually_reviewed") is not True):
        raise PackageError("CogAlpha browser and visual acceptance has not passed.")
    checks = report.get("checks")
    if not isinstance(checks, dict) or checks.get("errors") != []:
        raise PackageError("Browser acceptance is incomplete or contains errors.")
    for name in ("modelSwitch", "replicateSwitch", "costSwitch", "riskSwitch", "actualReportDownload"):
        if checks.get(name) is not True:
            raise PackageError(f"Browser acceptance is missing a passed interaction: {name}")
    if checks.get("materialDownloads") != 4:
        raise PackageError("Browser acceptance does not cover all four materials.")
    ui_hash = sha256(regular_file(root, "src/cfquant/ui_cogalpha.py").read_bytes())
    evidence_hash = sha256(regular_file(root, "evidence/project2_cogalpha/manifest.json").read_bytes())
    if report.get("ui_source_sha256") != ui_hash:
        raise PackageError("Browser acceptance belongs to a different UI source.")
    if report.get("evidence_manifest_sha256") != evidence_hash:
        raise PackageError("Browser acceptance belongs to a different evidence manifest.")
    return {"file": FINAL_UI, "sha256": sha256(path.read_bytes()), "passed": True,
            "ui_source_sha256": ui_hash, "evidence_manifest_sha256": evidence_hash,
            "screenshots_visually_reviewed": True}


def check_destination(destination: Path) -> tuple[Path, Path]:
    if destination.is_symlink():
        raise PackageError("Destination must be a real folder, not a symbolic link.")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise PackageError("Destination already exists with content; choose a new folder.")
    archive = destination.with_name(destination.name + ".zip")
    digest_file = destination.with_name(destination.name + ".zip.sha256")
    if archive.exists() or digest_file.exists():
        raise PackageError("Outer ZIP or its checksum already exists; choose a new destination.")
    return archive, digest_file


def regular_file(root: Path, relative: str) -> Path:
    name = relative_path(relative)
    path = root / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise PackageError(f"Missing or unsafe required file: {name}")
    return path


def validate_manifest(root: Path, study_key: str) -> dict:
    prefix = f"evidence/{study_key}"
    path = regular_file(root, prefix + "/manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise PackageError(f"Final evidence manifest is empty: {study_key}")
    for relative, expected in files.items():
        relative = relative_path(relative)
        if not isinstance(expected, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected):
            raise PackageError(f"Invalid evidence hash: {study_key}/{relative}")
        actual = regular_file(root, prefix + "/" + relative)
        content = actual.read_bytes()
        if sha256(content) != expected.lower():
            raise PackageError(f"Evidence hash mismatch: {study_key}/{relative}")
        findings = learned_model_findings(prefix + "/" + relative, content)
        if findings:
            raise PackageError(f"Public fitted-model payload rejected: {study_key}/{relative}: {', '.join(findings)}")
    for name in ("protocol.json", "source_reference.json", "validation.json", "performance.csv"):
        if name not in files:
            raise PackageError(f"Final evidence does not seal {name}: {study_key}")
    if study_key == "project2_cogalpha" and "model_metadata.json" not in files:
        raise PackageError("CogAlpha final evidence must include sanitized model_metadata.json.")
    validation = json.loads(regular_file(root, prefix + "/validation.json").read_text(encoding="utf-8"))
    if validation.get("all_ledgers_passed") is not True:
        raise PackageError(f"Study ledgers are not validated: {study_key}")
    count = validation.get("main_runs", validation.get("portfolio_runs"))
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise PackageError(f"Study has no completed investment experiments: {study_key}")
    return {"files_checked": len(files), "manifest_sha256": sha256(path.read_bytes()),
            "all_ledgers_passed": True, "validated_runs": count}


def deliverable_files(root: Path, directory: str) -> list[str]:
    folder = root / directory
    if not folder.is_dir() or folder.is_symlink():
        raise PackageError(f"Missing delivery directory: {directory}")
    files = []
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise PackageError(f"Symlink in delivery directory: {directory}")
        if not path.is_file():
            continue
        if path.suffix.lower() in INTERMEDIATE_SUFFIXES or path.name.endswith(("_compile.txt", ".synctex.gz")):
            continue
        files.append(path.relative_to(root).as_posix())
    return files


def check_material(path: Path) -> None:
    if path.stat().st_size == 0:
        raise PackageError(f"Empty material: {path.name}")
    if path.suffix.lower() == ".pdf":
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise PackageError(f"Invalid PDF header: {path.name}")
    elif path.suffix.lower() == ".pptx":
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None or "ppt/presentation.xml" not in archive.namelist():
                    raise PackageError(f"Invalid PPTX structure: {path.name}")
        except zipfile.BadZipFile:
            raise PackageError(f"Invalid PPTX ZIP: {path.name}") from None


def validate_source_archive(content: bytes, root: Path, required: set[str]) -> dict:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if archive.testzip() is not None:
                raise PackageError("Source ZIP CRC check failed.")
            names = set()
            normalized_names = set()
            for entry in archive.infolist():
                entry_path = relative_path(entry.filename.rstrip("/"))
                if not public_path(entry_path):
                    raise PackageError(f"Private or unsafe source archive path: {entry_path}")
                if entry.is_dir():
                    continue
                name = entry_path
                if name.casefold() in normalized_names:
                    raise PackageError(f"Duplicate/case-colliding archive entry: {name}")
                normalized_names.add(name.casefold())
                names.add(name)
                if stat.S_ISLNK(entry.external_attr >> 16) or not public_path(name):
                    raise PackageError(f"Private or unsafe source archive path: {name}")
                file_content = archive.read(entry)
                findings = credential_findings(name, file_content)
                if findings:
                    raise PackageError(f"Credential check rejected {name}: {', '.join(findings)}")
                findings = learned_model_findings(name, file_content)
                if findings:
                    raise PackageError(f"Public fitted-model payload rejected: {name}: {', '.join(findings)}")
            missing = sorted(required - names)
            if missing:
                raise PackageError("Required files are not in committed HEAD: " + ", ".join(missing[:12]))
            for name in sorted(required):
                if sha256(archive.read(name)) != sha256(regular_file(root, name).read_bytes()):
                    raise PackageError(f"Committed source differs from checked file: {name}")
            for spec in STUDIES.values():
                prefix = f"evidence/{spec['key']}/"
                manifest = json.loads(archive.read(prefix + "manifest.json"))
                for relative, expected in manifest["files"].items():
                    if sha256(archive.read(prefix + relative)) != expected.lower():
                        raise PackageError(f"Archived evidence hash mismatch: {prefix}{relative}")
            return {"archive_crc_passed": True, "entries": len(names),
                    "required_files_matched_committed_head": len(required),
                    "private_path_check_passed": True, "credential_literal_findings": 0,
                    "public_cogalpha_fitted_weight_fields_absent": True,
                    "credential_scan_scope": "text files; private key blocks, likely live key literals and sensitive paths"}
    except zipfile.BadZipFile:
        raise PackageError("Invalid source ZIP.") from None


def preflight(root: Path, destination: Path, release: str) -> dict:
    root = root.resolve()
    if not re.fullmatch(r"v\d+\.\d+\.\d+", release):
        raise PackageError("Release must be a semantic label such as v2.4.0.")
    check_destination(destination)
    if git(root, "status", "--porcelain=v1", "-z", "--untracked-files=no").strip():
        raise PackageError("Tracked working tree/index is dirty. Commit the intended release before packaging.")
    revision = git(root, "rev-parse", "HEAD").decode("ascii").strip()
    if not re.fullmatch(r"[a-f0-9]{40,64}", revision):
        raise PackageError("Cannot identify a committed HEAD.")
    version = tomllib.loads(regular_file(root, "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if release != "v" + version:
        raise PackageError("Release label does not match pyproject.toml version.")
    required = {"README.md", "pyproject.toml", "app.py", "requirements-lock.txt", "requirements-research-lock.txt",
                "scripts/package_project2_submission.py", "scripts/run_project2_rsrs.py", "scripts/run_project2_cogalpha.py",
                AUDIT_SCRIPT, FINAL_AUDIT, FINAL_UI, ERRATUM}
    required.update("src/cfquant/" + name for name in (
        "rsrs.py", "rsrs_study.py", "rsrs_diagnostics.py", "ui_rsrs.py", "cogalpha.py",
        "cogalpha_models.py", "cogalpha_mining.py", "cogalpha_study.py", "ui_cogalpha.py",
        "research_api.py", "factor_lab.py", "engine.py", "analytics.py", "data.py"))
    for directory, pattern in (("scripts", "*project2*"), ("tests", "test_rsrs*.py"),
                               ("tests", "test_cogalpha*.py"), ("tests", "test_research_api.py")):
        required.update(path.relative_to(root).as_posix() for path in (root / directory).glob(pattern) if path.is_file())
    required.update("docs/" + name for name in COMMON_DOCS)
    release_files = deliverable_files(root, "evidence/project2_release")
    if "evidence/project2_release/test_acceptance.json" not in release_files:
        raise PackageError("Missing published Project 2 test acceptance record.")
    required.update(release_files)
    validation = {}
    study_files = {}
    for title, spec in STUDIES.items():
        key = spec["key"]
        validation[title] = validate_manifest(root, key)
        config = f"configs/{key}.json"
        config_path = regular_file(root, config)
        manifest = json.loads((root / f"evidence/{key}/manifest.json").read_text(encoding="utf-8"))
        expected_config = manifest.get("config_sha256", manifest.get("protocol_sha256"))
        if expected_config != sha256(config_path.read_bytes()):
            raise PackageError(f"Current config does not match sealed study identity: {key}")
        protocol = json.loads((root / f"evidence/{key}/protocol.json").read_text(encoding="utf-8"))
        if protocol != json.loads(config_path.read_text(encoding="utf-8")):
            raise PackageError(f"Evidence protocol and current config disagree: {key}")
        required.add(config)
        for name in (spec["audit"], spec["script"]):
            required.add("docs/" + name)
        prefix = f"reports/{key}/"
        for suffix in MATERIALS:
            name = prefix + spec["prefix"] + "_" + suffix
            check_material(regular_file(root, name))
            required.add(name)
        for name in TEX_FILES:
            source = regular_file(root, prefix + name).read_text(encoding="utf-8")
            for student_id, member in MEMBERS:
                if student_id not in source or member not in source:
                    raise PackageError(f"Missing team identity in {prefix}{name}")
            required.add(prefix + name)
        figures = deliverable_files(root, prefix + "figures")
        if not figures:
            raise PackageError(f"No report figures: {key}")
        study_files[title] = deliverable_files(root, f"reports/{key}") + deliverable_files(root, f"evidence/{key}")
        required.update(study_files[title])
    for path in required:
        regular_file(root, path)
    erratum_validation = validate_source_erratum(root)
    audit_validation = validate_release_audit(root)
    ui_validation = validate_ui_acceptance(root)
    # Archive the immutable commit, not a working tree or a zip of local folders.
    content = git(root, "archive", "--format=zip", revision)
    source_validation = validate_source_archive(content, root, required)
    # Catch concurrent edits/commits before any destination writes.
    if git(root, "status", "--porcelain=v1", "-z", "--untracked-files=no").strip() or git(root, "rev-parse", "HEAD").decode().strip() != revision:
        raise PackageError("Repository changed during preflight; retry after the release is stable.")
    return {"revision": revision, "release": release, "archive": content,
            "evidence": validation, "source_validation": source_validation,
            "source_erratum": erratum_validation, "formal_audit": audit_validation,
            "browser_acceptance": ui_validation,
            "study_files": study_files, "release_files": release_files}


def write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def validate_outer_doc_links(destination: Path) -> dict:
    """Check shared Markdown documentation against the actual delivery layout."""
    checked = 0
    root = destination.resolve()
    for path in sorted((root / "docs").glob("*.md")):
        for raw_target in re.findall(r"\]\(([^)\n]+)\)", path.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            resolved = (path.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(root) or not resolved.exists():
                raise PackageError(f"Broken local delivery link in {path.relative_to(root).as_posix()}: {target}")
            checked += 1
    return {"passed": True, "local_links_checked": checked,
            "scope": "outer docs/*.md; relative Markdown links in the actual package folder"}


def readme(release: str, revision: str, source_name: str) -> str:
    team = "；".join(f"{number} {name}" for number, name in MEMBERS)
    return f"""# 青序 Project 2：两份独立研究提交材料

小组：{team}
发布标签：{release}（标签文字不证明远程 Git tag 已创建）
代码提交：{revision}
代码仓库：{REPOSITORY_URL}
提交代码：{REPOSITORY_URL}/tree/{revision}

RSRS 与 CogAlpha 分别提供研究报告、Beamer PDF、可编辑 PPTX、对应演讲稿 PDF/Markdown、LaTeX 源码、图表和公开聚合证据。它们分别回答不同研究问题，不应混用数据、收益或原图表。两个子目录顶部的四件文件便于直接打开。

每个子目录也保留 reports/project2_*/ 与 evidence/project2_*/ 的原工程布局。编辑 LaTeX 时请进入对应 reports/project2_* 目录，保留 figures 和 ../../evidence 的相对关系；不要只移动一个 tex 文件。三份TeX可在具备CTeX/Beamer与字体的XeLaTeX环境中独立编译，不依赖模型API或artifact运行环境。PPTX可直接在兼容软件中编辑，但源码重建需要 `@oai/artifact-tool` 和配套artifact运行时，不是仅安装Python依赖即可完成。共享源码包 `{source_name}` 保存完整已提交工程及构建脚本；不能从ZIP校验通过推断新电脑已成功编译。

先读 docs/PROJECT2_REQUIREMENTS_MATRIX.md 与 docs/PROJECT2_REPRODUCIBILITY.md。课件要求、研究补充假设和本地改造分别说明。数据快照、协议、代码/结果哈希及各自账本核验见研究证据；SHA256.json 校验交付文件，旁边的 .zip.sha256 校验整个压缩包。

公开审计入口是 `scripts/audit_project2_cogalpha.py`：`--self-test`为10项合成自检，`--final`才核对完成的私有/公开研究记录，不调用模型、不重跑策略。正式通过回执在 `evidence/project2_release/cogalpha_release_audit.json`；打包已要求其通过且审计脚本哈希匹配。实际浏览器/截图验收回执在同目录的 `cogalpha_ui_acceptance.json`，对应本版界面源码和最终证据清单。软件462项通过的汇总跨主研究/独立Qlib环境及重跑记录，不表示一次不中断全套测试0跳过，审计器自检另计。

公开源码与证据不包含原始授权行情、Tushare Token、模型密钥、拟合系数/截距、模型权重或付费调用缓存。Ridge系数和逐股预测留在私有运行目录；当前没有保存LightGBM完整树结构。CogAlpha离线evaluate无需Key或API，可用保存的已接受公式重建缓存，再按固定参数重新拟合；仍需有权访问的私有快照、完整提案记录和原预算摘要。生成新提案才涉及外部模型，本轮唯一账本为 `D:/Desktop/cf2026-quant-platform/runs/project2_cogalpha_api/`，共享50元上限；复制的离线目录或API账本不构成第二份额度。50元是预算而不是必须花完的金额。

`deepseek-flash`是API别名，当日官方映射不是可固定的权重检查点。保存公式的离线复算与以后实时API重新抽样是不同操作，后者不保证同结果。CogAlpha的沪深300基准是无手续费价格指数，股票组合则以复权研究单位计收益并扣费，不能将这种比较声称为同口径的全收益超额表现。详见复现说明。

离线复跑按说明先准备新的 `runs/reproduction_cogalpha/`，复制挖掘身份/公式记录和预算摘要，不复制API付费账本，不覆盖发布证据。PPTX部分时序曲线为可编辑展示抽样到约140点并保留末点；绩效指标、执行和公开日频CSV没有抽样。

两项实验都是已观察历史上的研究。RSRS原始Wind样本不可得，CogAlpha使用不同样本、静态股票池、服务模型及部分替代估计器，属于受限机制复现，不声称原表精确复现，也不承诺未来盈利。亏损和未改善结果保留。

原文链接及本机PDF位置见 来源与本机数据说明.md。完整第三方原文PDF未放入公开源码包；教师需要原文时，可使用合法来源链接或用户已有的本机文件另行提供。本包尚未代交 Blackboard，不代表课程系统已接收或教师评分。
"""


def source_notes() -> str:
    return """# 来源与本机数据说明

## 课程

用户提供：`D:/Desktop/游戏 启动/CF2026_Project2.pdf`，8页正式要求。
SHA-256：`346219417ef143471a8461102dfeef06e1560a1eecd08698f320ad2582fc1ad2`。
截止日期2026-11-01、约15分钟演示来自用户确认；课件没有规定两篇都必须完成。

## RSRS

光大证券，刘均伟、周萧潇，2017-05-01，《基于阻力支撑相对强度（RSRS）的市场择时》。
[固定版本来源目录](https://github.com/hugo2046/QuantsPlaybook/tree/3f83a77f4c7996ed8185b8979a42c4372c1d811f/C-%E6%8B%A9%E6%97%B6%E7%B1%BB/RSRS%E6%8B%A9%E6%97%B6%E6%8C%87%E6%A0%87)。
本机原文：`D:/Desktop/CF2026_Project2_研报复现/sources/Everbright_RSRS_20170501.pdf`。
论文来源、页码和已知歧义见RSRS源审计与source_reference.json；公开证据仅保留带来源的研究对照片段，不分发完整研报。

## CogAlpha

[arXiv:2511.18850v4](https://arxiv.org/abs/2511.18850v4)，Cognitive Alpha Mining via LLM-Driven Code-Based Evolution。
[正式出版记录](https://aclanthology.org/2026.acl-long.538/)。
本机审计副本：`D:/Desktop/2511.18850v4.pdf`；用户早期文件位置为 `D:/Desktop/游戏 启动/2511.18850v4.pdf`。
真实论文文件SHA-256以CogAlpha子目录中 `configs/project2_cogalpha_source_erratum.json` 和源审计为准。冻结协议曾把论文SHA误录为65字符；独立勘误提供正确64字符身份，并保留冻结配置原字节身份，不改变计算或重新调用模型。公开source_reference和最终清单分别引用正确来源身份与勘误文件哈希。

## 私有数据

RSRS默认快照目录：当前代码根目录下 `data/private/project2_rsrs/`。
本机Tushare数据根：`D:/Desktop/cf2026-quant-platform`；所需市场、特征缓存、基准、日历和依赖输入详见复现说明。公开Git和本提交源码ZIP均不含这些原始授权行情，也不含任何凭据。
CogAlpha完整私有运行位于该根目录的 `runs/project2_cogalpha_formal/`，唯一API预算账本位于 `runs/project2_cogalpha_api/`；从构建目录移动后已逐文件核对SHA-256，原位置不再保留第二份可继续付费账本。它们不在本提交包中。本机迁移回执为 `runs/project2_private_migration.json`。
这里的本机路径是原交付机器的说明，不保证其他电脑存在相同路径；不要把路径文本当成数据文件本身。
"""


def package(root: Path, destination: Path, release: str, *, check_only: bool = False) -> dict:
    plan = preflight(root, destination, release)
    summary = {key: plan[key] for key in ("revision", "release", "evidence", "source_validation", "source_erratum", "formal_audit", "browser_acceptance")}
    if check_only:
        return {**summary, "check_only": True, "destination_created": False}
    outer_zip, outer_checksum = check_destination(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source_name = f"CF2026_Project2_Source_{release}.zip"
    write_new(destination / source_name, plan["archive"])
    # Copy from the validated immutable archive, never from files another worker
    # could change between the check and the copy.
    with zipfile.ZipFile(io.BytesIO(plan["archive"])) as archive:
        for title, spec in STUDIES.items():
            key = spec["key"]
            for name in plan["study_files"][title]:
                write_new(destination / title / name, archive.read(name))
            for suffix in MATERIALS:
                filename = spec["prefix"] + "_" + suffix
                write_new(destination / title / filename, archive.read(f"reports/{key}/{filename}"))
            for filename in (spec["audit"], spec["script"]):
                path = "docs/" + filename
                write_new(destination / title / path, archive.read(path))
                write_new(destination / path, archive.read(path))
            path = f"configs/{key}.json"
            write_new(destination / title / path, archive.read(path))
            write_new(destination / path, archive.read(path))
            if title == "CogAlpha":
                write_new(destination / title / ERRATUM, archive.read(ERRATUM))
                write_new(destination / ERRATUM, archive.read(ERRATUM))
        for name in COMMON_DOCS:
            write_new(destination / "docs" / name, archive.read("docs/" + name))
        for name in plan["release_files"]:
            write_new(destination / name, archive.read(name))
    write_new(destination / "先读我.md", readme(release, plan["revision"], source_name).encode("utf-8"))
    write_new(destination / "来源与本机数据说明.md", source_notes().encode("utf-8"))
    write_new(destination / "CODE_COMMIT.txt", (plan["revision"] + "\n").encode("ascii"))
    doc_links = validate_outer_doc_links(destination)
    receipt = {"schema": "qingxu.project2.submission.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               **summary, "members": [{"student_id": number, "name": name} for number, name in MEMBERS],
               "raw_vendor_data_included": False, "blackboard_submitted": False,
               "outer_document_links": doc_links,
               "files": {p.relative_to(destination).as_posix(): sha256(p.read_bytes())
                         for p in sorted(destination.rglob("*")) if p.is_file()}}
    write_new(destination / "SHA256.json", (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    with zipfile.ZipFile(outer_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, destination.name + "/" + path.relative_to(destination).as_posix())
    with zipfile.ZipFile(outer_zip) as archive:
        if archive.testzip() is not None:
            raise PackageError("Outer ZIP CRC failed; partial output retained for inspection.")
        for relative, expected in receipt["files"].items():
            if sha256(archive.read(destination.name + "/" + relative)) != expected:
                raise PackageError("Outer ZIP content differs from its delivery manifest.")
    digest = sha256(outer_zip.read_bytes())
    write_new(outer_checksum, (digest + "  " + outer_zip.name + "\n").encode("utf-8"))
    return {**summary, "destination": str(destination), "outer_zip": str(outer_zip),
            "outer_zip_crc_passed": True, "outer_zip_sha256": digest,
            "outer_document_links": doc_links,
            "outer_zip_checksum": str(outer_checksum), "delivery_files": len(receipt["files"]) + 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True, help="new empty delivery folder; sibling ZIP is also created")
    parser.add_argument("--release", default="v2.4.0")
    parser.add_argument("--check-only", action="store_true", help="validate committed source and materials without creating output")
    args = parser.parse_args()
    try:
        result = package(ROOT, args.destination.resolve(), args.release, check_only=args.check_only)
    except (PackageError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        # No deletion, rollback or overwrite on failure.
        print(f"Packaging stopped: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
