from distutils.core import setup
from distutils.command.clean import clean
from distutils.command.install import install

requirements = [
    'scikit-learn==0.23.2',
    'numpy==1.20.0',
    'scipy==1.7.1'
]


class CleanInstall(install):
    # Calls the default run command, then deletes the build area
    # (equivalent to "setup clean --all").
    def run(self):
        install.run(self)
        c = clean(self.distribution)
        c.all = True
        c.finalize_options()
        c.run()


setup(
    name="git_cluster",
    version="1.0",
    author="gaozhangyang",
    author_email="gaozhangyang@westlake.edu.cn",
    description="GIT: Clustering Based on Graph of Intensity Topology",
    packages=['git_cluster'],
    install_requires=requirements,
    # egg_base='/tmp',
    cmdclass={'install': CleanInstall}
)