# Generated Outputs

Training, evaluation, and inference create content here at runtime, including:

```text
outputs/
|-- checkpoints/
|-- predictions/
|-- reports/
`-- sota/
```

Generated files are excluded from Git. This prevents trained weights, large
visualizations, local filesystem paths, and machine-specific reports from
entering the source repository. The main README records the aggregate metrics
from the completed experiments.

For a public model release, publish weights separately as a versioned GitHub
Release or model-registry artifact only after confirming that the training-data
licenses permit redistribution. Include the exact config, commit, evaluation
metrics, and a model card with every released checkpoint.
