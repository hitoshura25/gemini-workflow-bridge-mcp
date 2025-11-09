from setuptools import setup, find_packages
import os


def local_scheme(version):
    if os.environ.get("IS_PULL_REQUEST"):
        return f".dev{os.environ.get('GITHUB_RUN_ID', 'local')}"
    return ""


try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""


setup(
    name='hitoshura25-gemini-workflow-bridge',
    author='Vinayak Menon',
    author_email='hitoshura.25@gmail.com',
    description='MCP server that bridges Claude Code to Gemini CLI for workflow tasks like codebase analysis, specification creation, and code review',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Vinayak Menon/hitoshura25-gemini-workflow-bridge',
    use_scm_version={"local_scheme": local_scheme},
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add your runtime dependencies here
    ],
    python_requires='>=3.11',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        'console_scripts': [
            'hitoshura25-gemini-workflow-bridge=hitoshura25_gemini_workflow_bridge.cli:main',
            'mcp-hitoshura25-gemini-workflow-bridge=hitoshura25_gemini_workflow_bridge.server:main',
        ],
    },
)