# Open Targets Croissant exporter 🥐

This is the Open Targets [Croissant](https://docs.mlcommons.org/croissant/docs/croissant-spec.html)
exporter. It generates a Croissant-compliant dataset for the Open Targets data
pipeline.




## Running

To run the dockerize version:

```
docker build --tag 'ot_croissant' .
docker run -it 'ot_croissant'
```

```
usage: ot_croissant [-h] --output OUTPUT -d DATASET

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Output file path
  -d DATASET, --dataset DATASET
                        Dataset to include
```

## Development

> [!IMPORTANT]
> Remember to run `make dev` in the root of the repository before starting development.
> This will set up a pre-commit checks and install dependencies.
