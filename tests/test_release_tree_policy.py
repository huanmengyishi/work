from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CURRENT_RELEASE_NOTE = Path("v0.13.0.md")
PERMITTED_DOC_FILES = {Path("releases") / CURRENT_RELEASE_NOTE}
LOCAL_ONLY_ROOTS = (
    Path("历史资料"),
    Path("本地开发资料"),
    Path("项目运行审计与改进建议"),
    Path("测试与验收"),
    Path("老版使用说明"),
    Path("老版工作日志"),
)
REQUIRED_APPLICATION_ASSETS = (
    Path("pyproject.toml"),
    Path("requirements.txt"),
    Path(".github/workflows/test.yml"),
    Path("agent/__init__.py"),
    Path("agent/cli.py"),
    Path("tests/conftest.py"),
    Path("scripts/build_release_docx.py"),
    Path("launcher/agent"),
)


def test_detailed_local_material_stays_out_of_repository_tree() -> None:
    unexpected = [path for path in LOCAL_ONLY_ROOTS if (REPOSITORY_ROOT / path).exists()]

    assert unexpected == [], (
        "Detailed development, audit, test-data, and historical materials must stay "
        f"in the local documentation workspace, not GitHub: {unexpected}"
    )


def test_release_notes_contain_only_current_release() -> None:
    docs_root = REPOSITORY_ROOT / "docs"
    published_docs = {path.relative_to(docs_root) for path in docs_root.rglob("*") if path.is_file()}

    assert published_docs == PERMITTED_DOC_FILES, (
        "GitHub must publish exactly docs/releases/v0.13.0.md; architecture reports, "
        "implementation notes, gap analyses, and historical release notes belong in "
        "the verified local archive, while an empty docs tree is not a valid release: "
        f"missing={sorted(PERMITTED_DOC_FILES - published_docs)}, "
        f"unexpected={sorted(published_docs - PERMITTED_DOC_FILES)}"
    )


def test_release_tree_keeps_application_delivery_assets() -> None:
    missing = [path for path in REQUIRED_APPLICATION_ASSETS if not (REPOSITORY_ROOT / path).is_file()]

    assert missing == [], (
        "Release cleanup must retain application source, tests, CI, scripts, launcher, "
        f"and packaging metadata: {missing}"
    )
