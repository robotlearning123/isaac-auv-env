# Security Policy

## Reporting a Vulnerability

Open a private [GitHub Security Advisory](https://github.com/robotlearning123/oceanscale/security/advisories/new).

Do **not** file security issues in public GitHub issues.

We aim to:
- Acknowledge within 72 hours.
- Provide a remediation timeline within 7 days.
- Coordinate disclosure with the reporter.

## Supported Versions

OceanScale is in pre-release (v0.0.x). The v0.5.x tag history reflects pre-positioning website iterations and is not supported.

| Version | Supported |
|---------|-----------|
| 0.0.x   | yes (latest) |
| 0.5.x   | no (legacy)  |

## Scope

**In scope:**
- Code in `oceanscale/` and `website/`
- Build pipelines in `.github/workflows/`
- Dependency vulnerabilities (Python + npm)

**Out of scope:**
- Third-party services (Cloudflare Pages, PyPI, etc.) — report to vendor directly
- Documentation-only issues (use a regular bug report)
