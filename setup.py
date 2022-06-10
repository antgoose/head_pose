import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="head_pose_package_antgoose",
    version="0.0.1",
    author="Anton Gusev",
    author_email="asgusev_1@edu.hse.ru",
    description="It's a demo of CV project",
    long_description="This package generates executable that performs head pose detection",
    long_description_content_type="text/markdown",
    url="https://github.com/antgoose/head_pose",
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)