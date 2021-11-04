from setuptools import setup

requirements = [
    'scikit-learn==0.23.2',
    'numpy==1.19.2',
    'scipy==1.7.1'
]

setup(
    name="git_cluster",
    version="1.0",
    author="gaozhangyang",
    author_email="gaozhangyang@westlake.edu.cn",
    description="GIT: Clustering Based on Graph of Intensity Topology",
    packages=['git_cluster'],
    install_requires=requirements
)