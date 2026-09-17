# Bool Memory Probe

Base: `submissions/candidate_v4_plus8_task149_bool_onnx`
Output: `submissions/bool_memory_probe_onnx`
Kept rewrites: 3/3
Total score gain: 0.491842
Total measured cost save: 11473

| task | kept | cost save | score gain | local pass | exact decoded train/test/arc-gen | rewrite | error |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 333 | yes | 1680 | 0.069874 | 265/265 | 3/3/1/1/261/261 | replace final float one-hot Sum/Concat with bool logic, then Cast before opset11 Pad |  |
| 374 | yes | 4399 | 0.335320 | 267/267 | 4/4/1/1/262/262 | bool ch0/ch1/ch2/ch4 concat; Pad bool out10_b |  |
| 301 | yes | 5394 | 0.086648 | 266/266 | 3/3/1/1/262/262 | replace final OneHot/Where with bool Equal tail index against color ids |  |
