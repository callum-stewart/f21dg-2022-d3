#!/bin/sh
docker run --rm \
	   -v ${PWD}:/data \
	   --user $(id -u):$(id -g) \
	   pandoc/core README.md -o output.pdf

