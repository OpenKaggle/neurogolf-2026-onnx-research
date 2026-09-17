# Bool Memory Probe

Base: `submissions/candidate_v4_plus8_task149_bool_onnx`
Output: `submissions/bool_memory_probe_onnx`
Kept rewrites: 6/6
Total score gain: 0.910498
Total measured cost save: 19463

| task | kept | cost save | score gain | local pass | exact decoded train/test/arc-gen | rewrite | error |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 333 | yes | 1680 | 0.069874 | 265/265 | train 3/3; test 1/1; arc-gen 261/261 | replace final float one-hot Sum/Concat with bool logic, then Cast before opset11 Pad |  |
| 374 | yes | 4399 | 0.335320 | 267/267 | train 4/4; test 1/1; arc-gen 262/262 | bool ch0/ch1/ch2/ch4 concat; Pad bool out10_b |  |
| 301 | yes | 5394 | 0.086648 | 266/266 | train 3/3; test 1/1; arc-gen 262/262 | replace final OneHot/Where with bool Equal tail index against color ids |  |
| 316 | yes | 360 | 0.096772 | 266/266 | train 3/3; test 1/1; arc-gen 262/262 | drop Cast(onehot_b->float); Pad bool onehot_b directly |  |
| 308 | yes | 1960 | 0.094214 | 266/266 | train 3/3; test 1/1; arc-gen 262/262 | drop Cast(out7_b->float); Pad bool out7_b directly |  |
| 240 | yes | 5670 | 0.227670 | 266/266 | train 3/3; test 1/1; arc-gen 262/262 | convert base_bg/updates to bool and ScatterND bool output |  |
