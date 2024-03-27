# How to contribute

Contributions are always welcome. Everything ranging from small extensions of the
documentation to implementing new features is appreciated. Of course, the bigger the
change the more it is necessary to reach out to us in advance for an discussion. You can
post an issue or contact [janosg](https://github.com/janosg) via email.

To get acquainted with the code base, you can also check out our
[issue tracker](https://github.com/OpenSourceEconomics/estimagic/issues) for some
immediate and clearly defined tasks.

1. Assuming you have settled on contributing a small fix to the project, please read the
   {ref}`style_guide` on the next page before you continue.

1. Next, fork the [repository](https://github.com/OpenSourceEconomics/estimagic/). This
   will create a copy of the repository where you have write access. Your fix will be
   implemented in your copy. After that, you will start a pull request (PR) which means
   a proposal to merge your changes into the project. If you plan to become a regular
   contributor we can give you push access to unprotected branches, which makes the
   process more convenient for you.

1. Clone the repository to your disk. Set up the project environment with conda and the
   and install your local version of estimagic in editable mode. The commands for this
   are (in a terminal in the root of your local estimagic repo):

   `conda env create -f environment.yml`

   `conda activate estimagic`

   `pip install -e .`

1. Implement the fix or new feature.

1. We validate contributions in three ways. First, we have a test suite to check the
   implementation of estimagic. Second, we correct for stylistic errors in code and
   documentation using linters. Third, we test whether the documentation builds
   successfully.

   You can run the test suit with

   ```bash
   $ pytest
   ```

   This will run the complete test suite. To run only a subset of the suite you can use
   the environments, `pytest`, `linting` and `sphinx`, with the `-e` flag of tox.

   Correct any errors displayed in the terminal.

   To correct stylistic errors, you can also install the linters as a pre-commit with

   ```bash
   $ pre-commit install
   ```

   Then, all the linters are executed before each commit and the commit is aborted if
   one of the check fails. You can also manually run the linters with

   ```bash
   $ pre-commit run -a
   ```

   Lastly, check if the documentation builds correctly. To do so, go to the root
   directory of your local estimagic repo and move to the `docs` folder. There, you need
   to create the `estimagic-rtd` environment (I you haven't done so already) and
   activate it:

   ```bash
   $ conda env create -f environment.yml
   ```

   ```bash
   $ conda activate estimagic-rtd`
   ```

   Still in the docs folder, now build the sphinx documentation by typing

   ```bash
   $ make html
   ```

   and sphinx will create the documentation pages for you. You can open the html pages
   of the documentation in you browser (e.g. Google Chrome) via

   ```bash
   $ google-chrome build/html/index.html
   ```

   and check if everything looks fine. Instead of the index page, you can also open a
   specific documentation page, e.g.,

   ```bash
   $ google-chrome build/html/explanations/optimization/why_optimization_is_hard.html
   ```

   Whatever page you are on, you can always move to another section by simply clicking -
   exactly as you would do in the online documentation.

5) If the tests pass, push your changes to your repository. Go to the Github page of
   your fork. A banner will be displayed asking you whether you would like to create a
   PR. Follow the link and the instructions of the PR template. Fill out the PR form to
   inform everyone else on what you are trying to accomplish and how you did it.

   The PR also starts a complete run of the test suite on a continuous integration
   server. The status of the tests is shown in the PR. Reiterate on your changes until
   the tests pass on the remote machine.

1) Ask one of the main contributors to review your changes. Include their remarks in
   your changes.

1) The final PR will be merged by one of the main contributors.
