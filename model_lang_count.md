# Model Language Distribution Statistics

| Model | NULL | arabic | chinese | control | digits | english | greek | hebrew | japanese | korean | pure_english | russian | thai | Total |
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek_v3 | 6456 | 3185 | 35184 | 1011 | 1110 | 49324 | 616 | 622 | 880 | 1120 | 21994 | 5251 | 1247 | 128000 |
| qwen2.5-72b | 11374 | 3634 | 24966 | 3488 | 10 | 65279 | 127 | 3160 | 2064 | 3473 | 27376 | 4122 | 2570 | 151643 |
| MiniCPM3-4B | 6843 | 38 | 28322 | 4 | 10 | 21836 | 143 | 10 | 140 | 92 | 15832 | 158 | 12 | 73440 |
| internlm | 5141 | 13111 | 10364 | 0 | 10 | 44979 | 71 | 10 | 35 | 6394 | 23291 | 16082 | 9081 | 128569 |
| gpt-4o | 23900 | 8036 | 7449 | 3381 | 1110 | 95471 | 1498 | 2373 | 806 | 2365 | 37839 | 14211 | 1561 | 200000 |

## Summary

### deepseek_v3

- english: 49324
- chinese: 35184
- pure_english: 21994
- NULL: 6456
- russian: 5251
- arabic: 3185
- thai: 1247
- korean: 1120
- digits: 1110
- control: 1011
- japanese: 880
- hebrew: 622
- greek: 616

### qwen2.5-72b

- english: 65279
- pure_english: 27376
- chinese: 24966
- NULL: 11374
- russian: 4122
- arabic: 3634
- control: 3488
- korean: 3473
- hebrew: 3160
- thai: 2570
- japanese: 2064
- greek: 127
- digits: 10

### MiniCPM3-4B

- chinese: 28322
- english: 21836
- pure_english: 15832
- NULL: 6843
- russian: 158
- greek: 143
- japanese: 140
- korean: 92
- arabic: 38
- thai: 12
- digits: 10
- hebrew: 10
- control: 4

### internlm

- english: 44979
- pure_english: 23291
- russian: 16082
- arabic: 13111
- chinese: 10364
- thai: 9081
- korean: 6394
- NULL: 5141
- greek: 71
- japanese: 35
- digits: 10
- hebrew: 10

### gpt-4o

- english: 95471
- pure_english: 37839
- NULL: 23900
- russian: 14211
- arabic: 8036
- chinese: 7449
- control: 3381
- hebrew: 2373
- korean: 2365
- thai: 1561
- greek: 1498
- digits: 1110
- japanese: 806

