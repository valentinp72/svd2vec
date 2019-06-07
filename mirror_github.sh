#!/bin/sh

cd /tmp
git clone --bare git@git-lium.univ-lemans.fr:vpelloin/svd2vec.git
cd svd2vec.git
git push --mirror git@github.com:valentinp72/svd2vec.git
cd ..
rm -rf svd2vec.git
git clone git@github.com:valentinp72/svd2vec.git
cd svd2vec
git subtree push --prefix docs/build/html origin gh-pages
cd ..
rm -rf svd2vec
