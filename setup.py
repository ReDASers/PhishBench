from setuptools import setup, find_packages

import src.phishbench as phishbench

entry_points = {
    "console_scripts": ["phishbench=phishbench_cli.main:main",
                        "make-phishbench-config=phishbench_cli.update:main"]
}

with open("requirements.txt") as req:
    reqs = req.readlines()

setup(name='phishbench',
      version=phishbench.__version__,
      description='A Phishing detection benchmarking framework',
      url='https://github.com/ReDASers/Phishing-Detection',
      packages=find_packages("src"),
      package_dir={"": "src"},
      package_data={"": ["alexa-top-1m.csv"]},
      install_requires=reqs,
      entry_points=entry_points)
