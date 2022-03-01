import setuptools

from utils import get_version

with open('requirements.txt') as f:
    reqs = [line for line in f]
reqs.append('pytest-spec')

setuptools.setup(name="slaid",
                 version=get_version(),
                 description="AI for automatic analysis of slides",
                 long_description_content_type="text/markdown",
                 packages=setuptools.find_packages(),
                 classifiers=[
                     "Programming Language :: Python :: 3",
                 ],
                 python_requires='>=3.6',
                 install_requires=reqs,
                 scripts=['bin/classify.py', 'bin/annotate_onnx.py'],
                 package_data={'': ['resources/models/*']},
                 include_package_data=True)
