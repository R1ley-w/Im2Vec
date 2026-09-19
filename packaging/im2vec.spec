# PyInstaller spec for the Im2Vec desktop app.
#
# Build from the repo root:
#     pyinstaller packaging/im2vec.spec --noconfirm
#
# Produces dist/Im2Vec.app on macOS, dist/Im2Vec.exe on Windows and
# dist/Im2Vec on Linux. PyInstaller cannot cross-compile, so each platform is
# built on that platform (see .github/workflows/desktop.yml).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent
NAME = "Im2Vec"

a = Analysis(
    [str(ROOT / "packaging" / "launch.py")],
    pathex=[str(ROOT)],
    datas=[(str(ROOT / "im2vec" / "web"), "im2vec/web")],
    # uvicorn picks its loop/protocol/lifespan implementations by string name.
    hiddenimports=collect_submodules("uvicorn"),
    # Dev and dataset tooling that must never end up in the bundle.
    excludes=["tkinter", "gradio", "scipy", "numpy", "cairosvg", "pyarrow", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

if sys.platform == "darwin":
    # A proper .app bundle: onedir inside Contents/, starts instantly.
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME,
              console=False, upx=False)
    coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)
    app = BUNDLE(
        coll,
        name=f"{NAME}.app",
        bundle_identifier="io.github.r1ley-w.im2vec",
        info_plist={
            "CFBundleDisplayName": NAME,
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
        },
    )
else:
    # One self-contained file: the simplest thing to download and run.
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name=NAME,
              console=False, upx=False)
