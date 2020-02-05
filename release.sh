#!/bin/bash

TAG=v`cat lore/__init__.py | grep  '__version__ ='  | awk '{ print $3}'`

git tag $TAG
git push origin $TAG

rm -r build/ dist/ ./*.egg-info/

python setup.py sdist bdist_wheel

twine upload -r pypi dist/*

