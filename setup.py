import setuptools

install_requires = [
    'pandas',
    'numpy>=1.15.4',
    'glob2>=0.6',
    'seaborn',
    'matplotlib',
    'imblearn',
    'times',
    'sklearn'
]



setuptools.setup(
    name="MyToolBox",
    version="0.0.1",
    author="Bruno Vinicius Nonato",
    author_email="brunovinicius154@gmail.com",
    description="A tool box to speed up projects",
    url="https://github.com/bruno154/project-mytoolbox",
    install_requires=install_requires,
    #python_requires='==3.8',
    packages=['MyToolBox']
)