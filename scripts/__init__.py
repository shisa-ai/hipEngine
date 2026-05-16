"""Importable helper namespace for tests.

Executable scripts still run by path; this file prevents third-party ``scripts``
packages on ``sys.path`` from shadowing repository helper modules during pytest.
"""
