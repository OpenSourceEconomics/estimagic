from pathlib import Path

SRC = Path(__file__).parent.parent.resolve() / "src"


to_append = r"""
import warnings
warnings.warn(
    "estimagic has been renamed to optimagic. Please uninstall estimagic and install "
    "optimagic instead. Don't worry, your estimagic imports will still work if you "
    "install optimagic, and simple warnings will help you to adjust them for future "
    "releases.\n\n"
    "To make these changes using pip, run:\n"
    "-------------------------------------\n"
    "$ pip uninstall estimagic\n"
    "$ pip install optimagic\n\n"
    "For conda users, use:\n"
    "---------------------\n"
    "$ conda remove estimagic\n"
    "$ conda install -c conda-forge optimagic\n",
    FutureWarning,
)
"""

for package in ("estimagic", "optimagic"):
    init_file = SRC / package / "__init__.py"
    current_content = init_file.read_text()
    new_content = current_content + to_append
    init_file.write_text(new_content)
