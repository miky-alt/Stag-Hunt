"""Installs the vendored (modified) Gymnasium-Stag-Hunt environments used by this project."""
from setuptools import find_packages, setup

setup(
    name="gymnasium_stag_hunt",
    version="0.0.2",
    description=(
        "Markov Stag Hunt gridworlds for Gymnasium (fork of giorgiofranceschelli/Gymnasium-Stag-Hunt) "
        "extended with per-agent behavioural event counters and a stag-mobility probability."
    ),
    url="https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt",
    packages=find_packages(include=["gymnasium_stag_hunt", "gymnasium_stag_hunt.*"]),
    include_package_data=True,
    package_data={"gymnasium_stag_hunt": ["assets/*", "assets/**/*"]},
    install_requires=["gymnasium", "pygame", "opencv-python", "pettingzoo"],
    python_requires=">=3.11",
)
