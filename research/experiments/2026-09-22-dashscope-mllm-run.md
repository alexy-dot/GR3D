# DashScope MLLM executions

- Date: 2026-09-22
- Scope: OSI scene 0000, hardened six-condition Phase-B package
- Protocol: v1 image-only 60-request run plus corrected v2 scene-context rerun
- Model: `qwen3-vl-32b-instruct` (requested and returned)
- Decoding: temperature 0, maximum 64 output tokens
- Compatible endpoint: `https://maas.qianwenaiapi.com/compatible-mode/v1`

## Protocol-v1 implementation and correction

The original request generator sent only images and questions. It did not serialize the `scene_instances` file declared by the two ID conditions. Consequently, the completed 60-request run below is an **image-only ID-render experiment**, not a test of `3D-only IDs + scene table`. Its files and hashes are preserved unchanged.

Protocol v2 corrects that omission. Only the two ID conditions receive a deterministic, answer-blind scene table containing scene ID, `S###`, semantic label/name, selected 3D geometry fields, evidence, and declared motion state/evidence. Representative crops are associated with their `S###` entry through the validated catalog. Raw, RGB and no-ID conditions receive no scene table. Neither protocol reads `ground_truth.json` during request preparation.

The runner now writes a `.run.json` sidecar with request-manifest hash, package-manifest hash, requested model, provider/base URL, decoding parameters and protocol version. Resume requires exact metadata equality and unique known request IDs. This prevents mixing predictions after a package, request manifest, endpoint, model or decoding change.

## Preserved protocol-v1 image-only run

The initial attempt exposed two operational issues: one Python request ended during TLS with `SSLEOFError`, and the next fresh login shell showed that the persisted credential was empty. After the credential was privately restored, a one-request smoke test succeeded. The resumable full run then completed all 60 unique requests without exposing ground truth to inference.

Total usage was 382,892 tokens: 382,678 prompt tokens and 214 completion tokens. The high prompt usage is dominated by images. Raw responses remain local and are intentionally excluded from Git.

| Condition | Count | Exact accuracy | Numerical MAE |
| --- | ---: | ---: | ---: |
| raw frames only | 10 | 0.10 | 3.4914 |
| raw + RGB canonical views | 10 | 0.10 | 7.3486 |
| raw + no-ID semantic views | 10 | 0.20 | 3.8343 |
| raw + 3D-only IDs | 10 | 0.10 | 4.1914 |
| raw + 3D-only IDs + representative crops | 10 | 0.10 | 4.2914 |
| tagged-question control | 10 | 0.20 | 4.4343 |

All conditions answered the trajectory-description question correctly. The no-ID semantic and tagged-control conditions each answered one of two relative-distance MCQs correctly; the other conditions answered neither. Numerical exact accuracy is deliberately strict, so MAE is the more informative diagnostic for numerical questions.

This is one scene, ten questions per condition, and project-local scoring rather than the official OSI evaluator. It does not demonstrate that no-ID semantics is generally superior. Because the scene table was absent, it also says nothing about whether structured `S###` geometry context helps.

## Artifact provenance

- Predictions: `/home/dell/project/GR3D/outputs/osi_0000_mllm_eval/qwen3_vl_32b_predictions.json`
- Predictions SHA-256: `d3d02e8213333d05aa351106fb030b309cbe39730a2d2698c5e44b24bd6496fa`
- Scores: `/home/dell/project/GR3D/outputs/osi_0000_mllm_eval/qwen3_vl_32b_scores.json`
- Scores SHA-256: `9a5b62b6e353cb936753af4c1230c712883d13f6ea14bb34b0430ea6d9c83743`

## Protocol-v2 scene-context rerun

The hardened package was revalidated before inference: 51 hashed files, 10 questions and 6 conditions. The regenerated protocol-v2 manifest contains 60 requests. An explicit audit found scene context in all 20 ID-condition requests and none of the 40 remaining requests; all 170 representative-crop image records carry a scene ID; serialized requests contain no `ground_truth` token.

Only the two affected ID conditions were rerun. The runner's strict sidecar check was exercised by resuming the second condition after 9 of 10 requests. All 20 unique requests completed with returned model `qwen3-vl-32b-instruct`.

Total v2 usage was 244,417 tokens: 244,342 prompt and 75 completion tokens.

| Condition | Count | Exact accuracy | Numerical MAE |
| --- | ---: | ---: | ---: |
| 3D-only IDs + scene table | 10 | 0.20 | 4.4629 |
| 3D-only IDs + scene table + representative crops | 10 | 0.10 | 4.5914 |

The scene-table condition answered one of two relative-distance MCQs and the trajectory-description item. The crop condition answered only the trajectory item. On this one scene, adding the scene table coincided with one more exact answer than the old image-only ID-render condition, while the crop condition did not improve. This is not evidence of a general scene-table advantage: the sample has only ten questions, scoring is project-local, the comparison is not an isolated text-only intervention, and the representative crops retain baked-in visual tags.

The complete reconstruction suite passed 78 tests. No additional scene, official OSI evaluator run or independent repetition was performed.

### Protocol-v2 artifact provenance

- Requests: `/home/dell/project/GR3D/outputs/osi_0000_mllm_eval/requests_protocol_v2_scene_context.json`
- Requests SHA-256: `87da229df1d520caf24590b5a1bafea97c7058f8b7ad08203e2c22516b30179d`
- Predictions: `/home/dell/project/GR3D/outputs/osi_0000_mllm_eval/qwen3_vl_32b_protocol_v2_scene_context_predictions.json`
- Predictions SHA-256: `33ad40afc38ea187ddc529270ae226b733c63e993f8d9ed11bb037675dddc672`
- Run sidecar SHA-256: `83faac5d4673a98542a378640fe477a137ff046ed247d1b40f5e4d597a61560d`
- Scores: `/home/dell/project/GR3D/outputs/osi_0000_mllm_eval/qwen3_vl_32b_protocol_v2_scene_context_scores.json`
- Scores SHA-256: `10e19caabe4fc0c6e83169dd82e733c99e85345333b975cb8ae30e67a5990069`
