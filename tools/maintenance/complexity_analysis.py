#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Complexity Analysis Automation
Project ID: Image Processing App 20251119
Author: The-Sage-Mage

Comprehensive complexity analysis with multiple metrics and automation.
"""

import subprocess
import sys
import json
import ast
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
import argparse
import statistics


class ComplexityMetric(NamedTuple):
    """Structure for complexity measurements."""
    file_path: str
    function_name: str
    line_number: int
    complexity: int
    complexity_type: str


class CodeComplexityAnalyzer:
    """Advanced code complexity analysis with multiple metrics."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.complexity_reports_dir = self.project_root / "complexity_reports"
        self.complexity_reports_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Complexity thresholds
        self.thresholds = {
            "cyclomatic": {
                "low": 5,
                "moderate": 10,
                "high": 15,
                "very_high": 20
            },
            "cognitive": {
                "low": 5,
                "moderate": 15,
                "high": 25,
                "very_high": 50
            },
            "halstead": {
                "low": 100,
                "moderate": 300,
                "high": 1000,
                "very_high": 3000
            }
        }
    
    def install_complexity_tools(self) -> bool:
        """Install complexity analysis tools."""
        print("?? Installing complexity analysis tools...")
        
        tools = [
            "radon",           # Cyclomatic complexity, raw metrics
            "mccabe",          # McCabe complexity
            "xenon",           # Code complexity monitoring
            "cognitive-complexity", # Cognitive complexity
            "wily",            # Complexity tracking over time
        ]
        
        for tool in tools:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", tool],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"   ? Installed {tool}")
            except subprocess.CalledProcessError as e:
                print(f"   ?? Failed to install {tool}: {e.stderr}")
        
        return True
    
    def analyze_cyclomatic_complexity(self, paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze cyclomatic complexity using radon."""
        if paths is None:
            paths = ["src/", "tests/"]
        
        print("?? Analyzing cyclomatic complexity...")
        
        results = {
            "tool": "radon",
            "metric_type": "cyclomatic_complexity",
            "functions": [],
            "summary": {},
            "timestamp": datetime.now().isoformat()
        }
        
        existing_paths = []
        for path in paths:
            path_obj = self.project_root / path
            if path_obj.exists():
                existing_paths.append(str(path_obj))
        
        if not existing_paths:
            return {"error": "No valid paths found"}
        
        try:
            # Run radon for cyclomatic complexity
            result = subprocess.run(
                [
                    sys.executable, "-m", "radon", "cc",
                    "--json",
                    "--show-complexity",
                ] + existing_paths,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                try:
                    radon_data = json.loads(result.stdout)
                    
                    all_functions = []
                    for file_path, functions in radon_data.items():
                        for func in functions:
                            complexity_metric = ComplexityMetric(
                                file_path=file_path,
                                function_name=func.get('name', ''),
                                line_number=func.get('lineno', 0),
                                complexity=func.get('complexity', 0),
                                complexity_type='cyclomatic'
                            )
                            all_functions.append(complexity_metric._asdict())
                    
                    results["functions"] = all_functions
                    
                    # Calculate summary statistics
                    if all_functions:
                        complexities = [f['complexity'] for f in all_functions]
                        results["summary"] = {
                            "total_functions": len(all_functions),
                            "average_complexity": round(statistics.mean(complexities), 2),
                            "median_complexity": statistics.median(complexities),
                            "max_complexity": max(complexities),
                            "min_complexity": min(complexities),
                            "high_complexity_count": len([c for c in complexities if c > self.thresholds["cyclomatic"]["high"]]),
                            "very_high_complexity_count": len([c for c in complexities if c > self.thresholds["cyclomatic"]["very_high"]])
                        }
                        
                        avg_complexity = results["summary"]["average_complexity"]
                        high_count = results["summary"]["high_complexity_count"]
                        
                        print(f"   ?? Average complexity: {avg_complexity}")
                        print(f"   ?? High complexity functions: {high_count}")
                        
                        # Show most complex functions
                        complex_functions = sorted(all_functions, key=lambda x: x['complexity'], reverse=True)[:5]
                        for func in complex_functions:
                            if func['complexity'] > self.thresholds["cyclomatic"]["moderate"]:
                                print(f"      {func['file_path']}:{func['line_number']} "
                                      f"{func['function_name']} (complexity: {func['complexity']})")
                    
                except json.JSONDecodeError as e:
                    print(f"   ? Failed to parse radon output: {e}")
                    results["error"] = "JSON parsing failed"
            
        except FileNotFoundError:
            print("   ? radon not found. Install with: pip install radon")
            return {"error": "radon not installed"}
        
        return results
    
    def analyze_cognitive_complexity(self, paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze cognitive complexity."""
        if paths is None:
            paths = ["src/"]  # Focus on main code for cognitive complexity
        
        print("?? Analyzing cognitive complexity...")
        
        results = {
            "tool": "cognitive_complexity",
            "metric_type": "cognitive_complexity", 
            "functions": [],
            "summary": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Use manual cognitive complexity calculation
        # (cognitive-complexity tool may not be available or reliable)
        try:
            all_functions = []
            
            for path_str in paths:
                path_obj = self.project_root / path_str
                if not path_obj.exists():
                    continue
                
                for py_file in path_obj.rglob("*.py"):
                    try:
                        cognitive_metrics = self._calculate_cognitive_complexity(py_file)
                        all_functions.extend(cognitive_metrics)
                    except Exception as e:
                        print(f"   ?? Error processing {py_file}: {e}")
            
            results["functions"] = [func._asdict() for func in all_functions]
            
            # Calculate summary
            if all_functions:
                complexities = [f.complexity for f in all_functions]
                results["summary"] = {
                    "total_functions": len(all_functions),
                    "average_complexity": round(statistics.mean(complexities), 2),
                    "median_complexity": statistics.median(complexities),
                    "max_complexity": max(complexities),
                    "min_complexity": min(complexities),
                    "high_complexity_count": len([c for c in complexities if c > self.thresholds["cognitive"]["high"]]),
                    "very_high_complexity_count": len([c for c in complexities if c > self.thresholds["cognitive"]["very_high"]])
                }
                
                avg_complexity = results["summary"]["average_complexity"]
                high_count = results["summary"]["high_complexity_count"]
                
                print(f"   ?? Average cognitive complexity: {avg_complexity}")
                print(f"   ?? High cognitive complexity functions: {high_count}")
                
                # Show most cognitively complex functions
                complex_functions = sorted(all_functions, key=lambda x: x.complexity, reverse=True)[:5]
                for func in complex_functions:
                    if func.complexity > self.thresholds["cognitive"]["moderate"]:
                        print(f"      {func.file_path}:{func.line_number} "
                              f"{func.function_name} (cognitive complexity: {func.complexity})")
            
        except Exception as e:
            print(f"   ? Cognitive complexity analysis failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _calculate_cognitive_complexity(self, file_path: Path) -> List[ComplexityMetric]:
        """Calculate cognitive complexity for functions in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            metrics = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = self._cognitive_complexity_for_function(node)
                    
                    metric = ComplexityMetric(
                        file_path=str(file_path.relative_to(self.project_root)),
                        function_name=node.name,
                        line_number=node.lineno,
                        complexity=complexity,
                        complexity_type='cognitive'
                    )
                    metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            print(f"   ?? Error parsing {file_path}: {e}")
            return []
    
    def _cognitive_complexity_for_function(self, func_node: ast.FunctionDef) -> int:
        """Calculate cognitive complexity for a single function."""
        complexity = 0
        nesting_level = 0
        
        def visit_node(node, current_nesting=0):
            nonlocal complexity
            
            # Increment for control flow structures
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                complexity += 1 + current_nesting
                current_nesting += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1 + current_nesting
                current_nesting += 1
            elif isinstance(node, (ast.Break, ast.Continue)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # AND/OR operators add complexity
                complexity += len(node.values) - 1
            
            # Recursively visit child nodes
            for child in ast.iter_child_nodes(node):
                visit_node(child, current_nesting)
        
        # Visit all nodes in the function
        for node in func_node.body:
            visit_node(node, 0)
        
        return complexity
    
    def analyze_halstead_metrics(self, paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze Halstead complexity metrics using radon."""
        if paths is None:
            paths = ["src/"]
        
        print("?? Analyzing Halstead complexity metrics...")
        
        results = {
            "tool": "radon_halstead",
            "metric_type": "halstead",
            "files": [],
            "summary": {},
            "timestamp": datetime.now().isoformat()
        }
        
        existing_paths = []
        for path in paths:
            path_obj = self.project_root / path
            if path_obj.exists():
                existing_paths.append(str(path_obj))
        
        try:
            # Run radon for Halstead metrics
            result = subprocess.run(
                [
                    sys.executable, "-m", "radon", "hal",
                    "--json",
                ] + existing_paths,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                try:
                    halstead_data = json.loads(result.stdout)
                    
                    all_files = []
                    for file_path, metrics in halstead_data.items():
                        if metrics:  # Skip empty results
                            file_metric = {
                                "file_path": file_path,
                                "volume": metrics.get('volume', 0),
                                "difficulty": metrics.get('difficulty', 0),
                                "effort": metrics.get('effort', 0),
                                "time": metrics.get('time', 0),
                                "bugs": metrics.get('bugs', 0),
                                "length": metrics.get('length', 0),
                                "vocabulary": metrics.get('vocabulary', 0)
                            }
                            all_files.append(file_metric)
                    
                    results["files"] = all_files
                    
                    # Calculate summary statistics
                    if all_files:
                        volumes = [f['volume'] for f in all_files if f['volume'] > 0]
                        efforts = [f['effort'] for f in all_files if f['effort'] > 0]
                        difficulties = [f['difficulty'] for f in all_files if f['difficulty'] > 0]
                        
                        if volumes:
                            results["summary"] = {
                                "total_files": len(all_files),
                                "average_volume": round(statistics.mean(volumes), 2),
                                "average_difficulty": round(statistics.mean(difficulties), 2) if difficulties else 0,
                                "average_effort": round(statistics.mean(efforts), 2) if efforts else 0,
                                "total_estimated_bugs": round(sum(f['bugs'] for f in all_files), 2),
                                "high_volume_files": len([v for v in volumes if v > self.thresholds["halstead"]["high"]]),
                                "very_high_volume_files": len([v for v in volumes if v > self.thresholds["halstead"]["very_high"]])
                            }
                            
                            avg_volume = results["summary"]["average_volume"]
                            est_bugs = results["summary"]["total_estimated_bugs"]
                            
                            print(f"   ?? Average volume: {avg_volume}")
                            print(f"   ?? Estimated bugs: {est_bugs}")
                            
                            # Show most complex files by volume
                            complex_files = sorted(all_files, key=lambda x: x['volume'], reverse=True)[:5]
                            for file_metric in complex_files:
                                if file_metric['volume'] > self.thresholds["halstead"]["moderate"]:
                                    print(f"      {file_metric['file_path']} "
                                          f"(volume: {file_metric['volume']:.0f}, "
                                          f"estimated bugs: {file_metric['bugs']:.2f})")
                    
                except json.JSONDecodeError as e:
                    print(f"   ? Failed to parse Halstead output: {e}")
                    results["error"] = "JSON parsing failed"
            
        except FileNotFoundError:
            print("   ? radon not found for Halstead metrics")
            return {"error": "radon not installed"}
        
        return results
    
    def analyze_raw_metrics(self, paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze raw code metrics (LOC, LLOC, SLOC, etc.) using radon."""
        if paths is None:
            paths = ["src/", "tests/"]
        
        print("?? Analyzing raw code metrics...")
        
        results = {
            "tool": "radon_raw",
            "metric_type": "raw_metrics",
            "files": [],
            "summary": {},
            "timestamp": datetime.now().isoformat()
        }
        
        existing_paths = []
        for path in paths:
            path_obj = self.project_root / path
            if path_obj.exists():
                existing_paths.append(str(path_obj))
        
        try:
            # Run radon for raw metrics
            result = subprocess.run(
                [
                    sys.executable, "-m", "radon", "raw",
                    "--json",
                ] + existing_paths,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                try:
                    raw_data = json.loads(result.stdout)
                    
                    all_files = []
                    for file_path, metrics in raw_data.items():
                        if metrics:
                            file_metric = {
                                "file_path": file_path,
                                "loc": metrics.get('loc', 0),      # Lines of Code
                                "lloc": metrics.get('lloc', 0),    # Logical Lines of Code  
                                "sloc": metrics.get('sloc', 0),    # Source Lines of Code
                                "comments": metrics.get('comments', 0),
                                "multi": metrics.get('multi', 0),  # Multi-line strings
                                "blank": metrics.get('blank', 0),  # Blank lines
                                "single_comments": metrics.get('single_comments', 0)
                            }
                            all_files.append(file_metric)
                    
                    results["files"] = all_files
                    
                    # Calculate summary statistics
                    if all_files:
                        total_loc = sum(f['loc'] for f in all_files)
                        total_lloc = sum(f['lloc'] for f in all_files)
                        total_comments = sum(f['comments'] for f in all_files)
                        total_blank = sum(f['blank'] for f in all_files)
                        
                        comment_ratio = (total_comments / total_lloc * 100) if total_lloc > 0 else 0
                        
                        results["summary"] = {
                            "total_files": len(all_files),
                            "total_loc": total_loc,
                            "total_lloc": total_lloc,
                            "total_comments": total_comments,
                            "total_blank_lines": total_blank,
                            "average_file_size": round(total_lloc / len(all_files), 1),
                            "comment_ratio": round(comment_ratio, 1),
                            "largest_file": max(all_files, key=lambda x: x['lloc'])['file_path'] if all_files else None,
                            "largest_file_size": max(f['lloc'] for f in all_files) if all_files else 0
                        }
                        
                        print(f"   ?? Total logical lines: {total_lloc}")
                        print(f"   ?? Comment ratio: {comment_ratio:.1f}%")
                        print(f"   ?? Average file size: {results['summary']['average_file_size']} lines")
                        
                        # Show largest files
                        large_files = sorted(all_files, key=lambda x: x['lloc'], reverse=True)[:3]
                        for file_metric in large_files:
                            if file_metric['lloc'] > 100:  # Show files over 100 logical lines
                                print(f"      {file_metric['file_path']} ({file_metric['lloc']} logical lines)")
                    
                except json.JSONDecodeError as e:
                    print(f"   ? Failed to parse raw metrics output: {e}")
                    results["error"] = "JSON parsing failed"
            
        except FileNotFoundError:
            print("   ? radon not found for raw metrics")
            return {"error": "radon not installed"}
        
        return results
    
    def run_comprehensive_complexity_analysis(self) -> Dict[str, Any]:
        """Run comprehensive complexity analysis."""
        print("?? Running comprehensive complexity analysis...")
        
        # Run all complexity analyses
        cyclomatic_results = self.analyze_cyclomatic_complexity()
        cognitive_results = self.analyze_cognitive_complexity()
        halstead_results = self.analyze_halstead_metrics()
        raw_metrics_results = self.analyze_raw_metrics()
        
        # Combine results
        comprehensive_report = {
            "timestamp": datetime.now().isoformat(),
            "project_path": str(self.project_root),
            "analyses": {
                "cyclomatic_complexity": cyclomatic_results,
                "cognitive_complexity": cognitive_results,
                "halstead_metrics": halstead_results,
                "raw_metrics": raw_metrics_results
            },
            "overall_assessment": {}
        }
        
        # Calculate overall complexity score
        complexity_score = self._calculate_overall_complexity_score(
            cyclomatic_results, cognitive_results, halstead_results, raw_metrics_results
        )
        
        comprehensive_report["overall_assessment"] = complexity_score
        
        # Save comprehensive report
        report_file = self.complexity_reports_dir / f"comprehensive_complexity_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        # Generate markdown summary
        self.generate_complexity_markdown_report(comprehensive_report)
        
        # Print summary
        print(f"\n?? COMPLEXITY ANALYSIS SUMMARY")
        print(f"   Overall Complexity Score: {complexity_score['score']}/100")
        print(f"   Complexity Grade: {complexity_score['grade']}")
        print(f"   Report saved: {report_file}")
        
        return comprehensive_report
    
    def _calculate_overall_complexity_score(self, cyclomatic: Dict, cognitive: Dict, 
                                          halstead: Dict, raw_metrics: Dict) -> Dict[str, Any]:
        """Calculate overall complexity score."""
        score = 100
        issues = []
        
        # Cyclomatic complexity penalties
        if "summary" in cyclomatic:
            avg_cyclomatic = cyclomatic["summary"].get("average_complexity", 0)
            high_cyclomatic = cyclomatic["summary"].get("high_complexity_count", 0)
            
            if avg_cyclomatic > self.thresholds["cyclomatic"]["moderate"]:
                penalty = min((avg_cyclomatic - self.thresholds["cyclomatic"]["moderate"]) * 2, 20)
                score -= penalty
                issues.append(f"High average cyclomatic complexity: {avg_cyclomatic}")
            
            if high_cyclomatic > 0:
                score -= min(high_cyclomatic * 5, 25)
                issues.append(f"{high_cyclomatic} functions with high cyclomatic complexity")
        
        # Cognitive complexity penalties
        if "summary" in cognitive:
            avg_cognitive = cognitive["summary"].get("average_complexity", 0)
            high_cognitive = cognitive["summary"].get("high_complexity_count", 0)
            
            if avg_cognitive > self.thresholds["cognitive"]["moderate"]:
                penalty = min((avg_cognitive - self.thresholds["cognitive"]["moderate"]) * 1.5, 20)
                score -= penalty
                issues.append(f"High average cognitive complexity: {avg_cognitive}")
            
            if high_cognitive > 0:
                score -= min(high_cognitive * 3, 20)
                issues.append(f"{high_cognitive} functions with high cognitive complexity")
        
        # Halstead complexity penalties
        if "summary" in halstead:
            est_bugs = halstead["summary"].get("total_estimated_bugs", 0)
            if est_bugs > 5:
                score -= min(est_bugs * 2, 15)
                issues.append(f"High estimated bug count: {est_bugs}")
        
        # File size penalties from raw metrics
        if "summary" in raw_metrics:
            avg_file_size = raw_metrics["summary"].get("average_file_size", 0)
            if avg_file_size > 200:
                penalty = min((avg_file_size - 200) * 0.1, 10)
                score -= penalty
                issues.append(f"Large average file size: {avg_file_size} lines")
            
            comment_ratio = raw_metrics["summary"].get("comment_ratio", 0)
            if comment_ratio < 10:
                score -= 10
                issues.append(f"Low comment ratio: {comment_ratio}%")
        
        # Determine grade
        final_score = max(0, round(score, 1))
        
        if final_score >= 90:
            grade = "A (Excellent)"
        elif final_score >= 80:
            grade = "B (Good)"
        elif final_score >= 70:
            grade = "C (Fair)"
        elif final_score >= 60:
            grade = "D (Poor)"
        else:
            grade = "F (Needs Major Refactoring)"
        
        return {
            "score": final_score,
            "grade": grade,
            "issues": issues,
            "recommendations": self._generate_complexity_recommendations(issues)
        }
    
    def _generate_complexity_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on complexity issues."""
        recommendations = []
        
        for issue in issues:
            if "cyclomatic complexity" in issue.lower():
                recommendations.append("Break down complex functions into smaller, focused functions")
                recommendations.append("Extract complex logic into separate helper methods")
            elif "cognitive complexity" in issue.lower():
                recommendations.append("Reduce nested loops and conditional statements")
                recommendations.append("Use early returns to reduce nesting levels")
            elif "bug count" in issue.lower():
                recommendations.append("Review code for potential logic errors")
                recommendations.append("Increase test coverage for complex functions")
            elif "file size" in issue.lower():
                recommendations.append("Split large files into smaller, focused modules")
                recommendations.append("Extract classes and functions into separate files")
            elif "comment ratio" in issue.lower():
                recommendations.append("Add docstrings and inline comments")
                recommendations.append("Document complex algorithms and business logic")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def generate_complexity_markdown_report(self, report: Dict) -> None:
        """Generate markdown complexity report."""
        md_file = self.complexity_reports_dir / f"complexity_summary_{self.timestamp}.md"
        
        with open(md_file, 'w') as f:
            f.write("# Code Complexity Analysis Report\n\n")
            f.write(f"**Generated:** {report['timestamp']}\n")
            
            assessment = report['overall_assessment']
            f.write(f"**Complexity Score:** {assessment['score']}/100\n")
            f.write(f"**Grade:** {assessment['grade']}\n\n")
            
            # Executive Summary
            score = assessment['score']
            if score >= 90:
                f.write("## ?? EXCELLENT - Low complexity, well-structured code\n\n")
            elif score >= 80:
                f.write("## ?? GOOD - Manageable complexity with minor issues\n\n")
            elif score >= 70:
                f.write("## ?? FAIR - Moderate complexity, some refactoring needed\n\n")
            elif score >= 60:
                f.write("## ?? POOR - High complexity, significant refactoring recommended\n\n")
            else:
                f.write("## ?? CRITICAL - Very high complexity, major refactoring required\n\n")
            
            # Detailed Analysis Results
            analyses = report['analyses']
            
            f.write("## Analysis Results\n\n")
            
            # Cyclomatic Complexity
            if 'cyclomatic_complexity' in analyses and 'summary' in analyses['cyclomatic_complexity']:
                cc_summary = analyses['cyclomatic_complexity']['summary']
                f.write("### Cyclomatic Complexity (Control Flow)\n")
                f.write(f"- **Average Complexity:** {cc_summary.get('average_complexity', 'N/A')}\n")
                f.write(f"- **Functions Analyzed:** {cc_summary.get('total_functions', 'N/A')}\n")
                f.write(f"- **High Complexity Functions:** {cc_summary.get('high_complexity_count', 'N/A')}\n")
                f.write(f"- **Maximum Complexity:** {cc_summary.get('max_complexity', 'N/A')}\n\n")
            
            # Cognitive Complexity
            if 'cognitive_complexity' in analyses and 'summary' in analyses['cognitive_complexity']:
                cog_summary = analyses['cognitive_complexity']['summary']
                f.write("### Cognitive Complexity (Mental Load)\n")
                f.write(f"- **Average Complexity:** {cog_summary.get('average_complexity', 'N/A')}\n")
                f.write(f"- **Functions Analyzed:** {cog_summary.get('total_functions', 'N/A')}\n")
                f.write(f"- **High Complexity Functions:** {cog_summary.get('high_complexity_count', 'N/A')}\n")
                f.write(f"- **Maximum Complexity:** {cog_summary.get('max_complexity', 'N/A')}\n\n")
            
            # Halstead Metrics
            if 'halstead_metrics' in analyses and 'summary' in analyses['halstead_metrics']:
                hal_summary = analyses['halstead_metrics']['summary']
                f.write("### Halstead Metrics (Software Science)\n")
                f.write(f"- **Average Volume:** {hal_summary.get('average_volume', 'N/A')}\n")
                f.write(f"- **Average Difficulty:** {hal_summary.get('average_difficulty', 'N/A')}\n")
                f.write(f"- **Estimated Bugs:** {hal_summary.get('total_estimated_bugs', 'N/A')}\n")
                f.write(f"- **Files Analyzed:** {hal_summary.get('total_files', 'N/A')}\n\n")
            
            # Raw Metrics
            if 'raw_metrics' in analyses and 'summary' in analyses['raw_metrics']:
                raw_summary = analyses['raw_metrics']['summary']
                f.write("### Code Metrics\n")
                f.write(f"- **Total Logical Lines:** {raw_summary.get('total_lloc', 'N/A')}\n")
                f.write(f"- **Average File Size:** {raw_summary.get('average_file_size', 'N/A')} lines\n")
                f.write(f"- **Comment Ratio:** {raw_summary.get('comment_ratio', 'N/A')}%\n")
                f.write(f"- **Files Analyzed:** {raw_summary.get('total_files', 'N/A')}\n\n")
            
            # Issues and Recommendations
            if assessment.get('issues'):
                f.write("## Issues Identified\n\n")
                for issue in assessment['issues']:
                    f.write(f"- {issue}\n")
                f.write("\n")
            
            if assessment.get('recommendations'):
                f.write("## Recommendations\n\n")
                for i, rec in enumerate(assessment['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
        
        print(f"?? Markdown report saved: {md_file}")


def main():
    """Main entry point for complexity analysis."""
    parser = argparse.ArgumentParser(description="Comprehensive code complexity analysis")
    parser.add_argument("--install", action="store_true", help="Install complexity tools")
    parser.add_argument("--cyclomatic", action="store_true", help="Analyze cyclomatic complexity")
    parser.add_argument("--cognitive", action="store_true", help="Analyze cognitive complexity")
    parser.add_argument("--halstead", action="store_true", help="Analyze Halstead metrics")
    parser.add_argument("--raw", action="store_true", help="Analyze raw code metrics")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    
    args = parser.parse_args()
    
    analyzer = CodeComplexityAnalyzer()
    
    if args.install:
        analyzer.install_complexity_tools()
    elif args.cyclomatic:
        analyzer.analyze_cyclomatic_complexity()
    elif args.cognitive:
        analyzer.analyze_cognitive_complexity()
    elif args.halstead:
        analyzer.analyze_halstead_metrics()
    elif args.raw:
        analyzer.analyze_raw_metrics()
    elif args.all or len(sys.argv) == 1:
        analyzer.run_comprehensive_complexity_analysis()


if __name__ == "__main__":
    main()