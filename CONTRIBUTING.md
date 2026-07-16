# Contributing Guidelines

Bug reports, feature requests, documentation corrections, and pull requests are
welcome. Check existing issues and pull requests before opening a duplicate.

When reporting a problem, include:

- reproducible steps;
- the revision being used;
- relevant local modifications;
- Python and AWS Region details;
- sanitized error output.

Before submitting a pull request:

1. Keep AWS account IDs, role ARNs, bucket names, credentials, and private URLs
   out of committed files.
2. Keep the change focused and avoid unrelated formatting or refactoring.
3. Preserve dry-run defaults for billable operations.
4. Add or update tests for behavior changes.
5. Run `python -m unittest discover -s tests -v`.
6. Document required quotas, permissions, cost, and cleanup.
7. Use a clear commit message and address CI failures.

By contributing, you agree that your contribution is licensed under the
repository's MIT No Attribution license. We may ask you to sign a
[Contributor License Agreement](https://en.wikipedia.org/wiki/Contributor_License_Agreement)
for larger changes.

This project follows the
[Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
Report potential security issues through the process in [SECURITY.md](SECURITY.md),
not through a public issue.
