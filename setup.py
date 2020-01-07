from setuptools import setup, find_packages

entry_points = {
    "console_scripts": ["phishbench=phishbench_cli.main:main",
                        "make-phishbench-config=phishbench_cli.update:main"]
}

setup(name='PhishBench',
      version='0.1',
      description='A Phishing detection benchmarking framework',
      url='https://github.com/sbaki2/Phishing-Detection',
      packages=find_packages("src"),
      package_dir={"": "src"},
      entry_points=entry_points)