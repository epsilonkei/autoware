#!/bin/sh
# -*- mode: sh; coding: utf-8-unix; -*-

GIT_MEMO=$1/git_memo
ENV_MEMO=$1/env_memo
# Git status
echo "--------------- git last commit: ---------------" >> ${GIT_MEMO}
git log --name-status -p HEAD^..HEAD >> ${GIT_MEMO}
echo "--------------- git diff:        ---------------" >> ${GIT_MEMO}
git diff >> ${GIT_MEMO}

# Environment variables status
env >> ${ENV_MEMO}
