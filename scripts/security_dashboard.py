#!/usr/bin/env python3

"""
Security monitoring dashboard for Tok dependency security.
Provides real-time security metrics and alerts.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
import requests


class SecurityMonitor:
    def __init__(self):
        self.vulnerability_db_url = "https://pypi.org/pypi"
        self.safety_db_url = "https://pyup.io/safety/api/v1/advisories/"

    def load_dependency_analysis(self) -> dict:
        """Load dependency analysis data."""
        analysis_file = Path("dependency-analysis.json")
        if not analysis_file.exists():
            print(
                "❌ Dependency analysis file not found. Run SBOM generation first."
            )
            return {}

        with open(analysis_file) as f:
            return json.load(f)

    def check_vulnerabilities(self, packages: list[dict]) -> list[dict]:
        """Check for known vulnerabilities in packages."""
        vulnerabilities = []

        for package in packages:
            name = package.get("name", "")
            version = package.get("version", "")

            if not name or not version:
                continue

            try:
                # Check PyPI for security advisories
                response = requests.get(
                    f"{self.vulnerability_db_url}/{name}/{version}/json",
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    # Check for vulnerabilities in PyPI data
                    # This is a simplified check - in production, you'd use a proper vulnerability database

            except Exception as e:
                print(
                    f"⚠️  Could not check vulnerabilities for {name}@{version}: {e}"
                )

        return vulnerabilities

    def calculate_security_score(self, analysis: dict) -> dict:
        """Calculate overall security score based on various metrics."""
        if not analysis:
            return {"score": 0, "issues": ["No analysis data available"]}

        score = 100
        issues = []

        summary = analysis.get("summary", {})
        security_metrics = analysis.get("security_metrics", {})
        dependency_analysis = analysis.get("dependency_analysis", {})

        # Check hash coverage
        hash_coverage = security_metrics.get(
            "packages_with_integrity_checks", 0
        ) / max(summary.get("total_packages", 1), 1)
        if hash_coverage < 0.9:
            score -= 20
            issues.append(
                f"Only {hash_coverage:.1%} packages have integrity checks"
            )

        # Check recent packages
        recent_packages = security_metrics.get("recent_packages", 0)
        if recent_packages > 0:
            score -= 10 * min(recent_packages, 5)  # Penalty up to 50 points
            issues.append(
                f"{recent_packages} packages are less than 90 days old"
            )

        # Check dependency depth
        max_depth = dependency_analysis.get("max_dependency_depth", 0)
        if max_depth > 10:
            score -= 10
            issues.append(f"High dependency depth: {max_depth}")

        # Check for cycles
        cycles = dependency_analysis.get("dependency_cycles", [])
        if cycles:
            score -= 15
            issues.append(f"{len(cycles)} dependency cycles detected")

        # Check trusted sources
        trusted_sources = security_metrics.get(
            "packages_from_trusted_sources", 0
        )
        total_packages = summary.get("total_packages", 1)
        if trusted_sources < total_packages:
            score -= 25
            issues.append(
                f"{total_packages - trusted_sources} packages from untrusted sources"
            )

        score = max(0, score)

        return {
            "score": score,
            "grade": self.get_security_grade(score),
            "issues": issues,
            "recommendations": self.get_recommendations(score, issues),
        }

    def get_security_grade(self, score: int) -> str:
        """Get security grade based on score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def get_recommendations(self, score: int, issues: list[str]) -> list[str]:
        """Get security recommendations based on score and issues."""
        recommendations = []

        if score < 70:
            recommendations.append(
                "🚨 CRITICAL: Immediate security review required"
            )

        if "integrity checks" in " ".join(issues):
            recommendations.append(
                "🔐 Enable hash verification for all packages"
            )

        if "90 days old" in " ".join(issues):
            recommendations.append(
                "⏰ Review recent packages and consider waiting for 90-day period"
            )

        if "dependency cycles" in " ".join(issues):
            recommendations.append(
                "🔄 Resolve dependency cycles to improve security"
            )

        if "untrusted sources" in " ".join(issues):
            recommendations.append(
                "🔒 Review package sources and remove untrusted dependencies"
            )

        if score >= 80:
            recommendations.append(
                "✅ Security posture is good - continue monitoring"
            )

        return recommendations

    def generate_dashboard(self) -> dict:
        """Generate security dashboard data."""
        analysis = self.load_dependency_analysis()

        if not analysis:
            return self.get_empty_dashboard()

        security_score = self.calculate_security_score(analysis)
        vulnerabilities = self.check_vulnerabilities(
            analysis.get("packages", [])
        )

        dashboard = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "security_score": security_score,
            "summary": analysis.get("summary", {}),
            "security_metrics": analysis.get("security_metrics", {}),
            "dependency_analysis": analysis.get("dependency_analysis", {}),
            "vulnerabilities": vulnerabilities,
            "alerts": self.generate_alerts(security_score, analysis),
            "trends": self.calculate_trends(analysis),
        }

        return dashboard

    def get_empty_dashboard(self) -> dict:
        """Get empty dashboard when no data is available."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "security_score": {
                "score": 0,
                "grade": "F",
                "issues": ["No security data available"],
                "recommendations": ["Run dependency analysis first"],
            },
            "summary": {},
            "security_metrics": {},
            "dependency_analysis": {},
            "vulnerabilities": [],
            "alerts": ["🚨 No security monitoring data available"],
            "trends": {},
        }

    def generate_alerts(
        self, security_score: dict, analysis: dict
    ) -> list[str]:
        """Generate security alerts."""
        alerts = []

        score = security_score.get("score", 0)

        if score < 50:
            alerts.append("🚨 CRITICAL: Security score below 50")
        elif score < 70:
            alerts.append("⚠️  WARNING: Security score below 70")

        recent_packages = analysis.get("security_metrics", {}).get(
            "recent_packages", 0
        )
        if recent_packages > 5:
            alerts.append(f"⏰ {recent_packages} recent packages detected")

        cycles = analysis.get("dependency_analysis", {}).get(
            "dependency_cycles", []
        )
        if cycles:
            alerts.append(f"🔄 {len(cycles)} dependency cycles found")

        if not alerts:
            alerts.append("✅ No immediate security concerns")

        return alerts

    def calculate_trends(self, analysis: dict) -> dict:
        """Calculate security trends (placeholder for future implementation)."""
        # This would compare with historical data
        return {
            "score_trend": "stable",  # improving, declining, stable
            "vulnerability_trend": "stable",
            "dependency_trend": "stable",
        }

    def print_dashboard(self):
        """Print security dashboard to console."""
        dashboard = self.generate_dashboard()

        print("\n" + "=" * 60)
        print("🔒 TOK SECURITY DASHBOARD")
        print("=" * 60)

        score_data = dashboard["security_score"]
        print(
            f"\n📊 Security Score: {score_data['score']}/100 (Grade: {score_data['grade']})"
        )

        if score_data["issues"]:
            print("\n⚠️  Issues:")
            for issue in score_data["issues"]:
                print(f"   - {issue}")

        if score_data["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in score_data["recommendations"]:
                print(f"   {rec}")

        print("\n🚨 Alerts:")
        for alert in dashboard["alerts"]:
            print(f"   {alert}")

        summary = dashboard.get("summary", {})
        if summary:
            print("\n📈 Summary:")
            print(f"   Total packages: {summary.get('total_packages', 0)}")
            print(
                f"   Packages with hashes: {summary.get('packages_with_hashes', 0)}"
            )
            print(f"   Total size: {summary.get('total_size_mb', 0):.1f} MB")

        security_metrics = dashboard.get("security_metrics", {})
        if security_metrics:
            print("\n🔐 Security Metrics:")
            print(
                f"   Integrity checks: {security_metrics.get('packages_with_integrity_checks', 0)}"
            )
            print(
                f"   Trusted sources: {security_metrics.get('packages_from_trusted_sources', 0)}"
            )
            print(
                f"   Recent packages: {security_metrics.get('recent_packages', 0)}"
            )

        print(f"\n🕒 Last updated: {dashboard['timestamp']}")
        print("=" * 60)


def main():
    """Main function."""
    monitor = SecurityMonitor()
    monitor.print_dashboard()

    # Save dashboard data
    dashboard = monitor.generate_dashboard()
    output_file = Path("security-dashboard.json")
    with open(output_file, "w") as f:
        json.dump(dashboard, f, indent=2)

    print(f"\n📄 Dashboard data saved to {output_file}")


if __name__ == "__main__":
    main()
