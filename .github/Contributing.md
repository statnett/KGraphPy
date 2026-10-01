= Contributing to the project

Thank you for your interest in contributing to this project! We value your contributions and strive to create an open and welcoming environment. This document outlines the guidelines for participating and ensuring a smooth collaboration process.

== Code of conduct
All contributors are expected to adhere to our project's Code of Conduct. Please read it before contributing:

- **link:CODE_OF_CONDUCT.adoc[Code of Conduct]**

== Contribution Guidelines

=== General Guidelines
- Follow our coding standards and best practices.
- Ensure that your changes align with the link:ReadMe.adoc[project’s goals] and link:Roadmap.adoc[roadmap].
- Open an issue before starting significant work to gather feedback.
- All code must be reviewed before merging.

=== British English Convention
- Use **British English** for all comments, documentation, and discussions. A list of spelling differences between British and American English can be found link:https://www.britishcouncilfoundation.id/en/english/articles/british-and-american-english[here].
- Maintain clear and concise documentation.

== Developer Certificate of Origin (DCO)
Before contributing, you must agree to the Developer Certificate of Origin (DCO). This ensures that all contributions are properly licensed and attributable:

- **link:DCO[Developer Certificate of Origin]**

== License
By contributing to this project, you agree that your contributions will be licensed under the project's license:

- **link:LICENSE[Project License]**

== Governance
The project follows a structured governance model. Refer to the governance document for more details on roles, responsibilities, and decision-making processes:

- **link:Governance.adoc[Governance]**

== Roadmap
To align contributions with the project's vision and future direction, please review our roadmap:

- **link:Roadmap.adoc[Project Roadmap]**

== Security
If you discover a security vulnerability, please follow the responsible disclosure process outlined in our security policy:

- **Security.adoc[Security Policy]**

== Support
For questions, issues, or general discussions, refer to the support guidelines:

- **Support.adoc[Support Guidelines]**

== Submitting Changes

1. **Fork the Repository**
2. **Create a New Branch** for your feature or bug fix.
3. **Commit Your Changes**. Individual commit messages are free-form, but descriptive messages are appreciated.
4. **Run Tests** to ensure your changes do not introduce issues (`uv sync` then `uv run pytest`).
5. **Open a Pull Request** with a **Conventional Commit title** and link to the related issue (see below).
6. **Engage in Review** by addressing feedback and suggestions.

=== Pull Request Titles
Pull requests are **squash-merged**, so the PR title becomes the single commit on `main`. A CI check requires the title to follow link:https://www.conventionalcommits.org/[Conventional Commits]:

----
<type>[optional scope][!]: <description>
----

[cols="1,3"]
|===
| Type | Effect on the next release

| `feat` | New feature. Bumps the minor version and is listed under *Features*.
| `fix` | Bug fix. Bumps the patch version and is listed under *Bug Fixes*.
| `perf`, `deps`, `docs`, `revert` | Bump the patch version and appear in the changelog.
| `refactor`, `test`, `build`, `ci`, `chore` | Hidden from the changelog.
| `!` after the type, e.g. `feat!:` | Breaking change. Bumps the minor version while below 1.0.0, the major version afterwards.
|===

Examples: `feat: add CIM JSON-LD serializer`, `fix(cimxml): escape colons in predicates`, `feat!: remove deprecated header API`.

== Release Process
Releases are fully automated with link:https://github.com/googleapis/release-please[release-please]:

1. Every merge to `main` updates an open **Release PR** that bumps the version in `pyproject.toml` and updates `CHANGELOG.md` from the PR titles.
2. The Release PR is opened by GitHub Actions, so CI does not start on it automatically. A maintainer **closes and reopens** it to run the required checks, then merges it. This creates a `vX.Y.Z` tag and GitHub Release.
3. The distributions are built from the tag, and after approval in the `pypi` environment, published to link:https://pypi.org/project/kgraphpy/[PyPI] using Trusted Publishing (no API tokens), with signed provenance attestations.

Do not edit the version in `pyproject.toml` or `CHANGELOG.md` by hand.

== Issues and Feature Requests
- Before opening a new issue, check existing issues to avoid duplicates.
- Clearly describe the issue or feature request with relevant details.
- Label the issue appropriately for better tracking.



