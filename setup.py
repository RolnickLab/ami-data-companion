from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="trapdata",
    version="0.1",
    description="Deskop app for viewing and procesing data from autonomous insect monitoring stations.",
    url="http://github.com/notbot/trapdata",
    author="Michael Bunsen",
    author_email="notbot@gmail.com",
    license="MIT",
    packages=["trapdata"],
    zip_safe=False,
    entry_points={"console_scripts": ["trapdata=trapdata.main:run"]},
)
