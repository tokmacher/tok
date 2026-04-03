# Security Review Checklist for Dependency Updates

## 📋 Overview

This checklist must be completed for all dependency updates before merging.

## 🔍 Package Information

- **Package Name**:
- **Current Version**:
- **New Version**:
- **Update Type**: [ ] Security Patch [ ] Feature Update [ ] Bug Fix [ ] Major Version
- **PR Link**:

## ⏰ Age Verification

- [ ] Package is at least 90 days old
- [ ] Release date verified:
- [ ] No recent security advisories in the last 90 days

## 🔐 Security Analysis

- [ ] Checked PyPI for security advisories
- [ ] Reviewed changelog for security changes
- [ ] Verified package integrity (hashes match)
- [ ] Confirmed package source is trusted (PyPI)
- [ ] Checked for known CVEs in new version
- [ ] Analyzed dependency tree for new sub-dependencies

## 🚦 Blocked Package Check

- [ ] Package not in blocked list
- [ ] No suspicious package name similarities
- [ ] Verified maintainer reputation

## 🧪 Testing Verification

- [ ] All unit tests pass with new version
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed
- [ ] No breaking changes identified

## 📊 Risk Assessment

- **Risk Level**: [ ] Low [ ] Medium [ ] High
- **Impact**: [ ] Minimal [ ] Moderate [ ] Significant
- **Justification**:

## 👥 Review Process

- [ ] Initial security review completed
- [ ] Second reviewer approval obtained
- [ ] Team lead notification sent
- [ ] Documentation updated (if needed)

## 📝 Notes and Concerns

-

## ✅ Final Approval

- [ ] Security review completed - approved
- [ ] Ready for merge

______________________________________________________________________

## 🔗 Helpful Resources

- [PyPI Security Advisories](https://pypi.org/security)
- [CVE Database](https://cve.mitre.org)
- [Safety DB](https://pyup.io/safety/)
- [GitHub Advisory Database](https://github.com/advisories)

## 📞 Emergency Contacts

- Security Team: <tokmacher@protonmail.com>
- Team Lead: [Contact info]
- Emergency Response: [Contact info]

______________________________________________________________________

**Reviewer Name**:
**Review Date**:
**Signature**:
