from setuptools import setup


with open('requirements.txt') as requires:
    setup(
        name='StockSense',
        version='0.1.0',
        description='Stock analysis web application',
        author='Danila Yashin',
        author_email='danila.yashin23@gmail.com',
        packages=['StockSense'],
        install_requires=[
            line[:-1] for line in requires.readlines()],
    )
