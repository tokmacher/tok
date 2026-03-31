#!/usr/bin/env python3

"""
Dependency tree analysis script for Tok security monitoring.
Analyzes dependency depth, transitive dependencies, and security metrics.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import requests

def parse_uv_lock() -> list:
    """Parse uv.lock file to extract package information."""
    lock_file = Path("uv.lock")
    if not lock_file.exists():
        print("❌ uv.lock file not found")
        sys.exit(1)
    
    packages = []
    current_package = {}
    
    with open(lock_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('name = '):
            if current_package:
                packages.append(current_package)
            current_package = {"name": line.split('=')[1].strip().strip('"')}
        elif line.startswith('version = ') and current_package:
            current_package["version"] = line.split('=')[1].strip().strip('"')
        elif line.startswith('hash = ') and current_package:
            current_package["hash"] = line.split('=')[1].strip().strip('"')
        elif line.startswith('url = ') and current_package:
            current_package["url"] = line.split('=')[1].strip().strip('"')
        elif line.startswith('dependencies = ') and current_package:
            # Parse dependencies list
            deps_str = line.split('=')[1].strip()
            if deps_str.startswith('[') and deps_str.endswith(']'):
                deps_str = deps_str[1:-1]
                deps = [dep.strip().strip('"') for dep in deps_str.split(',') if dep.strip()]
                current_package["dependencies"] = deps
    
    if current_package:
        packages.append(current_package)
    
    return packages

def get_package_security_data(package_name: str, version: str) -> dict:
    """Fetch security data for a package."""
    try:
        response = requests.get(
            f"https://pypi.org/pypi/{package_name}/{version}/json",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant security information
        security_info = {
            "upload_time": data.get("uploads", [{}])[0].get("upload_time"),
            "package_size": sum(
                f.get("size", 0) for f in data.get("releases", {}).get(version, [])
            ),
            "has_wheel": any(
                f.get("packagetype") == "bdist_wheel" 
                for f in data.get("releases", {}).get(version, [])
            ),
            "has_source": any(
                f.get("packagetype") == "sdist" 
                for f in data.get("releases", {}).get(version, [])
            )
        }
        
        return security_info
    except Exception as e:
        print(f"⚠️  Could not fetch security data for {package_name}@{version}: {e}")
        return {}

def analyze_dependency_tree(packages: list) -> dict:
    """Analyze the dependency tree for security metrics."""
    
    # Build dependency graph
    dependency_graph = defaultdict(set)
    package_info = {}
    
    for package in packages:
        name = package.get("name", "")
        version = package.get("version", "")
        
        if not name:
            continue
        
        package_info[name] = {
            "version": version,
            "hash": package.get("hash", ""),
            "url": package.get("url", ""),
            "dependencies": package.get("dependencies", []),
            **get_package_security_data(name, version)
        }
        
        for dep in package.get("dependencies", []):
            dependency_graph[name].add(dep)
    
    # Calculate metrics
    analysis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_packages": len(packages),
            "total_dependencies": sum(len(p.get("dependencies", [])) for p in packages),
            "packages_with_hashes": sum(1 for p in packages if p.get("hash")),
            "packages_with_wheels": sum(1 for p in packages if package_info.get(p.get("name", ""), {}).get("has_wheel")),
            "total_size_mb": sum(
                package_info.get(p.get("name", ""), {}).get("package_size", 0) 
                for p in packages
            ) / (1024 * 1024)
        },
        "security_metrics": {
            "packages_with_integrity_checks": sum(1 for p in packages if p.get("hash")),
            "packages_from_trusted_sources": sum(
                1 for p in packages 
                if "pypi.org" in p.get("url", "")
            ),
            "recent_packages": 0  # Will be calculated below
        },
        "dependency_analysis": {
            "max_dependency_depth": calculate_max_depth(dependency_graph),
            "packages_with_no_dependencies": sum(1 for p in packages if not p.get("dependencies")),
            "most_depended_upon": find_most_depended_upon(dependency_graph),
            "dependency_cycles": find_dependency_cycles(dependency_graph)
        },
        "package_details": package_info
    }
    
    # Calculate recent packages (less than 90 days)
    ninety_days_ago = datetime.now(timezone.utc).timestamp() - (90 * 24 * 3600)
    for package in packages:
        name = package.get("name", "")
        upload_time_str = package_info.get(name, {}).get("upload_time")
        if upload_time_str:
            try:
                upload_time = datetime.fromisoformat(upload_time_str.replace('Z', '+00:00')).timestamp()
                if upload_time > ninety_days_ago:
                    analysis["security_metrics"]["recent_packages"] += 1
            except:
                pass
    
    return analysis

def calculate_max_depth(graph: dict) -> int:
    """Calculate maximum dependency depth."""
    def get_depth(package, visited=None):
        if visited is None:
            visited = set()
        
        if package in visited:
            return 0  # Cycle detected
        
        visited.add(package)
        max_depth = 0
        
        for dep in graph.get(package, set()):
            max_depth = max(max_depth, 1 + get_depth(dep, visited.copy()))
        
        return max_depth
    
    return max(get_depth(pkg) for pkg in graph.keys())

def find_most_depended_upon(graph: dict) -> list:
    """Find packages that are depended upon by the most other packages."""
    dependency_count = Counter()
    
    for package, deps in graph.items():
        for dep in deps:
            dependency_count[dep] += 1
    
    return dependency_count.most_common(10)

def find_dependency_cycles(graph: dict) -> list:
    """Find dependency cycles in the graph."""
    cycles = []
    visited = set()
    rec_stack = set()
    
    def dfs(package, path):
        if package in rec_stack:
            # Found a cycle
            cycle_start = path.index(package)
            cycle = path[cycle_start:] + [package]
            cycles.append(cycle)
            return
        
        if package in visited:
            return
        
        visited.add(package)
        rec_stack.add(package)
        
        for dep in graph.get(package, set()):
            dfs(dep, path + [package])
        
        rec_stack.remove(package)
    
    for package in graph.keys():
        if package not in visited:
            dfs(package, [])
    
    return cycles

def main():
    """Main analysis function."""
    print("🔍 Analyzing dependency tree...")
    
    packages = parse_uv_lock()
    print(f"📦 Found {len(packages)} packages")
    
    analysis = analyze_dependency_tree(packages)
    
    # Save analysis
    output_file = Path("dependency-analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print(f"\n📊 Dependency Analysis Summary:")
    print(f"  Total packages: {analysis['summary']['total_packages']}")
    print(f"  Total dependencies: {analysis['summary']['total_dependencies']}")
    print(f"  Max dependency depth: {analysis['dependency_analysis']['max_dependency_depth']}")
    print(f"  Packages with hashes: {analysis['summary']['packages_with_hashes']}")
    print(f"  Recent packages (<90 days): {analysis['security_metrics']['recent_packages']}")
    print(f"  Dependency cycles: {len(analysis['dependency_analysis']['dependency_cycles'])}")
    
    if analysis['dependency_analysis']['dependency_cycles']:
        print(f"\n⚠️  Dependency cycles detected:")
        for cycle in analysis['dependency_analysis']['dependency_cycles'][:5]:
            print(f"  - {' -> '.join(cycle)}")
    
    print(f"\n✅ Analysis saved to {output_file}")

if __name__ == "__main__":
    main()
