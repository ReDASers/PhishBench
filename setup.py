from setuptools import setup, find_packages

entry_points = {
    "console_scripts": ["phishbench=phishbench_cli.main:main",
                        "make-phishbench-config=phishbench_cli.update:main"]
}

with open("requirements.txt") as req:
    reqs = req.readlines()

setup(name='PhishBench',
      version='0.1.1',
      description='A Phishing detection benchmarking framework',
      url='https://github.com/sbaki2/Phishing-Detection',
      packages=find_packages("src"),
      package_dir={"": "src"},
      package_data={"": ["chromedriver"]},
      install_requires=reqs,
      entry_points=entry_points)
