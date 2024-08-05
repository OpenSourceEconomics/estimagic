(how-to-contribute)=

# How to contribute

## 1. Intro

We welcome and greatly appreciate contributions of all forms and sizes! Whether it's
updating the documentation, adding small extensions, or implementing new features, every
effort is valued.

For substantial changes, please contact us in advance. This allows us to discuss your
ideas and guide the development process from the beginning. You can start a conversation
by posting an issue on GitHub or by emailing [janosg](https://github.com/janosg).

To get familiar with the codebase, we recommend checking out our
[issue tracker](https://github.com/OpenSourceEconomics/optimagic/issues) for some
immediate and clearly defined tasks.

## 2. Before you start

Once you've decided to contribute, please review the {ref}`style_guide` (see the next
page) to ensure your work aligns with the project's coding standards.

We manage new features through Pull Requests (PRs). Contributors work on their local
copy of optimagic, modifying and extending the codebase there, before opening a PR to
propose merging their changes into the main branch.

Regular contributors gain push access to unprotected branches, which simplifies the
contribution process (see Notes below).

## 3. Step-by-step guide

1. Fork the [optimagic repository](https://github.com/OpenSourceEconomics/optimagic/).
   This action creates a copy of the repository with write access for you.

```{note}
For regular contributors: **Clone** the [repository](https://github.com/OpenSourceEconomics/optimagic/) to your local machine and create a new branch for implementing your changes. You can push your branch directly to the remote optimagic repository and open a PR from there.
```

2. Clone your forked repository to your disk. This is where you'll make all your
   changes.

1. Open your terminal and execute the following commands from the root directory of your
   local optimagic repository:

   ```console
   $ conda env create -f environment.yml
   $ conda activate optimagic
   $ pre-commit install
   ```

   These commands install optimagic in editable mode and activate pre-commit hooks for
   linting and style formatting.

1. Implement your fix or feature. Use git to add, commit, and push your changes to the
   remote repository. For more on git and how to stage and commit your work, refer to
   these
   [online materials](https://effective-programming-practices.vercel.app/git/staging/objectives_materials.html).

1. Contributions are validated in two main ways. We run a comprehensive test suite to
   ensure compatibility with the existing codebase and employ
   [pre-commit hooks](https://effective-programming-practices.vercel.app/git/pre_commits/objectives_materials.html)
   to maintain quality and adherence to our style guidelines. Opening a PR (see
   paragraph 7 below) triggers optimagic's
   [Continuous Integration (CI)](https://docs.github.com/en/actions/automating-builds-and-tests/about-continuous-integration)
   workflow, which runs the full `pytest` suite, pre-commit hooks, and other checks on a
   remote server.

   You can also run the test suite locally for
   [debugging](https://effective-programming-practices.vercel.app/debugging/pdbp/objectives_materials.html):

   ```console
   $ pytest
   ```

   With pre-commit installed, linters run before each commit. Commits are rejected if
   any checks fail. Note that some linters may automatically fix errors by modifying the
   code in-place. Remember to re-stage the files after such modifications.

```{tip}
Skip the next paragraph if you haven't worked on the documentation.
```

6. Assuming you have updated the documentation, verify that it builds correctly. From
   the root directory of your local optimagic repo, navigate to the docs folder and set
   up the optimagic-docs environment:

   ```console
   $ conda env create -f rtd_environment.yml
   $ conda activate optimagic-docs
   ```

   Inside the `docs` folder, run:

   ```console
   $ make html
   ```

   This command builds the HTML documentation, saving all files in the `build/html`
   directory. You can view the documentation with your preferred web browser by opening
   `build/html/index.html` or any other file. Similar to the online documentation, you
   can navigate to different pages simply by clicking on the links.

1. Once all tests and pre-commit hooks pass locally, push your changes to your forked
   repository and create a pull request through GitHub: Go to the Github repository of
   your fork. A banner on your fork's GitHub repository will prompt you to open a PR.

   ```{note}
   Regular contributors with push access can directly push their local branch to the remote optimagic repository and initiate a PR from there.
   ```

   Follow the steps outlined in the optimagic
   [PR template](https://github.com/OpenSourceEconomics/optimagic/blob/main/.github/PULL_REQUEST_TEMPLATE/pull_request_template.md)
   to describe your contribution, the problem it addresses, and your proposed solution.

   Opening a PR initiates a complete CI run, including the `pytest` suite, linters, code
   coverage checks, doctests, and building the HTML documentation. Monitor the CI
   workflow status on your PR page and make necessary modifications to your code based
   on the results, iterating until all tests pass.

1. Request a review from one of the main contributors once all CI tests pass. Address
   any feedback or suggestions by making the necessary changes and committing them.

1. After your PR is approved, one of the main contributors will merge it into
   optimagic's main branch.
