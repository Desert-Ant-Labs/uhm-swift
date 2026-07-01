# Third-party notices

The `uhm` model bundle that this SDK downloads from `huggingface.co/desert-ant-labs/uhm` and runs on-device incorporates upstream pretrained backbones. Their licenses apply to those components; nothing in the Desert Ant Labs Source-Available License overrides them.

## ntu-spml/distilhubert

- **Project:** DistilHuBERT
- **Hub:** <https://huggingface.co/ntu-spml/distilhubert>
- **License:** Apache License 2.0
- **Use:** Backbone for the shipped `uhm` Core ML model. Fine-tuned for frame-level filler classification and converted to Core ML.

## facebook/hubert-base-ls960

- **Project:** HuBERT
- **Hub:** <https://huggingface.co/facebook/hubert-base-ls960>
- **License:** Apache License 2.0
- **Use:** Upstream teacher/base architecture lineage for DistilHuBERT.

Neither upstream component ships a separate `NOTICE` file beyond the LICENSE. These notices ship inside `huggingface.co/desert-ant-labs/uhm` alongside the model weights.
