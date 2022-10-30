# MMCls

MMEval 中的 [BaseMetric](mmeval.core.BaseMetric) 参照了 [mmengine.evaluator](https://mmengine.readthedocs.io/zh_CN/latest/api/generated/mmengine.evaluator.Evaluator.html#mmengine.evaluator.Evaluator) 模块的设计，在此基础上引入了分布式通信后端的组件，以满足多样的分布式通信库需求。

因此 MMEval 天然的支持基于 OpenMMLab 2.0 算法库的评测，在 OpenMMLab 2.0 算法库中使用 MMEval 的评测指标无需多做修改。

以在 MMCls 中使用 [mmeval.Accuracy](mmeval.metrics.Accuracy) 为例，只需要在 config 中配置好使用的 Metric 为 Accuracy 即可：

```python
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_evaluator = val_evaluator
```

MMEval 对 OpenMMLab 2.0 算法库评测的支持正在逐步完善中，已支持的评测指标可以在[支持矩阵](../get_started/support_matrix.md) 中查看。
