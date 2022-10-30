# MMCls

[BaseMetric](mmeval.core.BaseMetric) in `MMEval` follows the design of the [mmengine.evaluator](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.evaluator.Evaluator.html#mmengine.evaluator.Evaluator) module and introduces distributed communication backend to meet the needs of a diverse distributed communication library.

Therefore, `MMEval` naturally supports the evaluation based on OpenMMLab 2.0 algorithm library, and the evaluation metrics using MMEval in OpenMMLab 2.0 algorithm library need not be modified.

For example, use [mmeval.Accuracy](mmeval.metrics.Accuracy) in MMCls, just configure the Metric to be Accuracy in the config:

```python
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_evaluator = val_evaluator
```

MMEval's support for OpenMMLab 2.0 algorithm library is being gradually improved, and the supported metric can be viewed in the [support matrix](../get_started/support_matrix.md).
