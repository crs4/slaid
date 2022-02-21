# Slaid
[![Docker Image CI](https://github.com/crs4/slaid/actions/workflows/test-build-publish.yaml/badge.svg)](https://github.com/mdrio/slaid/actions/workflows/test-build-publish.yaml)

## Intro
Slaid is a library for applying DL models from DeepHealth project (https://deephealth-project.eu/) on WSI. 

## Installation
Prerequisites: 
 * conda
 * python >=3.6, <= 3.8 (tested on 3.8)
 * Installation of dependencies pyecvl, pyeddl with conda is recommended. Be sure pip on your path is the one that comes with conda.

Run:
```
python setup.py install
```

## Docker image build

Run:
```
make docker
```

## Usage
For slide classification, use the installed bin classify.py. Get help typing:

```
classify.py --help
```

Examples:
Extract tissue
```
classify.py -f tissue -m  slaid/resources/models/tissue_model-extract_tissue_eddl_1.1.bin -l 2 -o <OUTPUT_DIR> <SLIDE>
```

Classify tumor:
```
classify.py -f tissue -m  slaid/resources/models/tumor_model-classify_tumor_eddl_0.1.bin -l 2 -o <OUTPUT_DIR> <SLIDE>
```

### Run on Docker
Slaid is released as docker images, one for each DL model available. 
Example:
```
docker run  --rm -v $DIR/../data:/data    slaid:0.62.0-tissue_model-extract_tissue_eddl_1.1 -l 0 /data/$IMAGE --overwrite  -f tissue  -o/data

```

Tests data from https://openslide.cs.cmu.edu/download/openslide-testdata/Mirax/. Credits to Yves Sucaet.
