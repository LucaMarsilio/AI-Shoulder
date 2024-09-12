from setuptools import setup, find_packages

setup(
    author="Luca Marsilio",
    author_email='luca.marsilio@polimi.it',
    python_requires='>=3.7',
    name='SegMentor',
    version='0.1.0',
    url='https://github.com/LucaMarsilio/SegMentor.git',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow",
        "numpy",
    ]
)
