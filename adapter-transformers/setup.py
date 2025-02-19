# Copyright 2020 The AdapterHub Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

from setuptools import find_packages, setup


# NOTE: All setup logic is transferred & adapted from Transformer's setup.py.
# We try to follow their general layout wherever sensible.

_deps = [
    "accelerate>=0.20.3",
    "black==22.3",  # after updating to black 2023, also update Python version in pyproject.toml to 3.7
    "datasets!=2.5.0",
    "dill<0.3.5",
    "docutils==0.16.0",
    "evaluate>=0.2.0",
    "flake8>=3.8.3",
    "GitPython<3.1.19",
    "hf-doc-builder>=0.3.0",
    "isort>=5.5.4",
    "Jinja2==2.11.3",
    "nltk",
    "onnxruntime-tools>=1.4.2",
    "onnxruntime>=1.4.0",
    "parameterized",
    "pillow",
    "protobuf<=3.20.2",
    "psutil",
    "pytest",
    "pytest-subtests",
    "pytest-timeout",
    "pytest-xdist",
    "markupsafe==2.0.1",
    "myst-parser",
    "rjieba",
    "rouge-score!=0.0.7,!=0.0.8,!=0.1,!=0.1.1",
    "sacrebleu>=1.4.12,<2.0.0",
    "sacremoses",
    "safetensors>=0.2.1",
    "scikit-learn",
    "sentencepiece>=0.1.91,!=0.1.92",
    "sphinx-copybutton",
    "sphinx-markdown-tables",
    "sphinx-rtd-theme==0.4.3",  # sphinx-rtd-theme==0.5.0 introduced big changes in the style.
    "sphinx==3.2.1",
    "sphinxext-opengraph==0.4.1",
    "sphinx-intl",
    "sphinx-multiversion",
    "timeout-decorator",
    "torch>=1.10,!=1.12.0",
    "transformers==4.33.3",
    "beautifulsoup4",
]


# this is a lookup table with items like:
#
# tokenizers: "tokenizers==0.9.4"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}

extras["sklearn"] = deps_list("scikit-learn")

extras["torch"] = deps_list("torch", "accelerate")

extras["onnxruntime"] = deps_list("onnxruntime", "onnxruntime-tools")

extras["sentencepiece"] = deps_list("sentencepiece", "protobuf")
extras["testing"] = deps_list(
    "pytest",
    "pytest-xdist",
    "timeout-decorator",
    "parameterized",
    "psutil",
    "datasets",
    "dill",
    "evaluate",
    "pytest-timeout",
    "black",
    "sacrebleu",
    "rouge-score",
    "nltk",
    "GitPython",
    "hf-doc-builder",
    "protobuf",  # Can be removed once we can unpin protobuf
    "sacremoses",
    "rjieba",
    "safetensors",
    "beautifulsoup4",
    "pillow",
    "accelerate",
)

extras["quality"] = deps_list("black", "datasets", "isort", "flake8", "GitPython", "hf-doc-builder")

extras["docs"] = deps_list(
    "docutils",
    "Jinja2",
    "markupsafe",
    "myst-parser",
    "sphinx",
    "sphinx-markdown-tables",
    "sphinx-rtd-theme",
    "sphinx-copybutton",
    "sphinxext-opengraph",
    "sphinx-intl",
    "sphinx-multiversion",
)

extras["dev"] = (
    extras["testing"]
    + extras["torch"]
    + extras["sentencepiece"]
    + extras["quality"]
    + extras["docs"]
    + extras["sklearn"]
    + extras["onnxruntime"]
)

# when modifying the following list, make sure to update src/transformers/dependency_versions_check.py
install_requires = [
    deps["transformers"],
]

setup(
    name="adapters",
    version="0.0.0.dev20230825",
    author="The AdapterHub team and community contributors",
    author_email="pfeiffer@ukp.tu-darmstadt.de",
    description="A Unified Library for Parameter-Efficient and Modular Transfer Learning",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning transformer pytorch BERT adapters",
    license="Apache",
    url="https://github.com/adapter-hub/adapters",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    package_data={"transformers": ["*.cu", "*.cpp", "*.cuh", "*.h", "*.pyx"]},
    zip_safe=False,
    extras_require=extras,
    python_requires=">=3.8.0",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
