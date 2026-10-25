Planted rare-name recall. A random four-syllable name is introduced in one sentence near the start of a context, then repeated after the given number of filler bytes ("... Then everybody said hello to " followed by the name); the score is the BPB of the name after its first two letters, so a model that remembers the name can lower it. `ctrl` replaces the first mention with a different name, so it measures what the model does without the cue. `kept` is the share of the planted name's patches still in the cache when the repeat starts. Higher accuracy and lower bits per byte mean better recall. Trials are independent random names on validation text.

## blt_tinystories_entropy (40 trials per distance)

Distance 400 filler bytes. Full cache 9.243 bits per byte, accuracy 0.092; control 9.268 bits, accuracy 0.092.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 9.080 | 0.092 | 0.000 |
| recency | 32 | 9.233 | 0.088 | 0.000 |
| recency | 64 | 9.263 | 0.083 | 0.000 |
| random | 16 | 9.141 | 0.092 | 0.062 |
| random | 32 | 9.181 | 0.092 | 0.127 |
| random | 64 | 9.202 | 0.083 | 0.310 |
| entropy | 16 | 9.153 | 0.096 | 0.042 |
| entropy | 32 | 9.083 | 0.092 | 0.173 |
| entropy | 64 | 9.081 | 0.092 | 0.451 |
| hybrid | 16 | 9.153 | 0.096 | 0.042 |
| hybrid | 32 | 9.139 | 0.083 | 0.119 |
| hybrid | 64 | 9.185 | 0.088 | 0.277 |
| h2o | 16 | 9.107 | 0.100 | 0.000 |
| h2o | 32 | 9.179 | 0.096 | 0.035 |
| h2o | 64 | 9.272 | 0.096 | 0.138 |
| snap | 16 | 9.269 | 0.083 | 0.000 |
| snap | 32 | 9.375 | 0.088 | 0.000 |
| snap | 64 | 9.331 | 0.092 | 0.119 |

Distance 1200 filler bytes. Full cache 9.486 bits per byte, accuracy 0.075; control 9.494 bits, accuracy 0.079.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 9.405 | 0.075 | 0.000 |
| recency | 32 | 9.490 | 0.075 | 0.000 |
| recency | 64 | 9.547 | 0.075 | 0.000 |
| random | 16 | 9.440 | 0.067 | 0.005 |
| random | 32 | 9.431 | 0.063 | 0.089 |
| random | 64 | 9.489 | 0.063 | 0.106 |
| entropy | 16 | 9.426 | 0.071 | 0.005 |
| entropy | 32 | 9.401 | 0.058 | 0.038 |
| entropy | 64 | 9.355 | 0.071 | 0.176 |
| hybrid | 16 | 9.426 | 0.071 | 0.005 |
| hybrid | 32 | 9.460 | 0.075 | 0.005 |
| hybrid | 64 | 9.488 | 0.075 | 0.074 |
| h2o | 16 | 9.391 | 0.075 | 0.000 |
| h2o | 32 | 9.503 | 0.063 | 0.017 |
| h2o | 64 | 9.588 | 0.067 | 0.147 |
| snap | 16 | 9.552 | 0.071 | 0.000 |
| snap | 32 | 9.620 | 0.075 | 0.000 |
| snap | 64 | 9.741 | 0.075 | 0.000 |

Distance 3000 filler bytes. Full cache 9.008 bits per byte, accuracy 0.092; control 9.017 bits, accuracy 0.088.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 8.938 | 0.079 | 0.000 |
| recency | 32 | 8.994 | 0.083 | 0.000 |
| recency | 64 | 8.983 | 0.092 | 0.000 |
| random | 16 | 8.849 | 0.088 | 0.006 |
| random | 32 | 8.808 | 0.108 | 0.020 |
| random | 64 | 8.802 | 0.088 | 0.037 |
| entropy | 16 | 8.940 | 0.092 | 0.000 |
| entropy | 32 | 8.921 | 0.092 | 0.017 |
| entropy | 64 | 8.807 | 0.079 | 0.068 |
| hybrid | 16 | 8.940 | 0.092 | 0.000 |
| hybrid | 32 | 9.028 | 0.083 | 0.000 |
| hybrid | 64 | 9.031 | 0.088 | 0.022 |
| h2o | 16 | 8.853 | 0.088 | 0.000 |
| h2o | 32 | 8.921 | 0.088 | 0.025 |
| h2o | 64 | 9.013 | 0.092 | 0.080 |
| snap | 16 | 9.064 | 0.083 | 0.000 |
| snap | 32 | 9.028 | 0.092 | 0.000 |
| snap | 64 | 9.131 | 0.100 | 0.000 |

## blt_tinystories_entropy_s1 (40 trials per distance)

Distance 400 filler bytes. Full cache 9.042 bits per byte, accuracy 0.071; control 9.009 bits, accuracy 0.075.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 8.871 | 0.063 | 0.000 |
| recency | 32 | 8.997 | 0.071 | 0.000 |
| recency | 64 | 9.038 | 0.067 | 0.000 |
| random | 16 | 8.905 | 0.063 | 0.062 |
| random | 32 | 8.884 | 0.067 | 0.127 |
| random | 64 | 8.919 | 0.083 | 0.310 |
| entropy | 16 | 8.916 | 0.079 | 0.042 |
| entropy | 32 | 8.742 | 0.088 | 0.173 |
| entropy | 64 | 8.634 | 0.075 | 0.451 |
| hybrid | 16 | 8.916 | 0.079 | 0.042 |
| hybrid | 32 | 8.907 | 0.083 | 0.119 |
| hybrid | 64 | 8.905 | 0.088 | 0.277 |
| h2o | 16 | 8.949 | 0.067 | 0.000 |
| h2o | 32 | 9.015 | 0.079 | 0.060 |
| h2o | 64 | 9.124 | 0.067 | 0.097 |
| snap | 16 | 9.018 | 0.071 | 0.000 |
| snap | 32 | 9.162 | 0.063 | 0.000 |
| snap | 64 | 9.152 | 0.058 | 0.156 |

Distance 1200 filler bytes. Full cache 9.310 bits per byte, accuracy 0.063; control 9.308 bits, accuracy 0.058.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 9.029 | 0.071 | 0.000 |
| recency | 32 | 9.190 | 0.067 | 0.000 |
| recency | 64 | 9.279 | 0.063 | 0.000 |
| random | 16 | 8.975 | 0.079 | 0.005 |
| random | 32 | 9.087 | 0.075 | 0.089 |
| random | 64 | 9.130 | 0.067 | 0.106 |
| entropy | 16 | 9.142 | 0.075 | 0.005 |
| entropy | 32 | 9.061 | 0.071 | 0.038 |
| entropy | 64 | 8.914 | 0.067 | 0.176 |
| hybrid | 16 | 9.142 | 0.075 | 0.005 |
| hybrid | 32 | 9.166 | 0.075 | 0.005 |
| hybrid | 64 | 9.236 | 0.058 | 0.074 |
| h2o | 16 | 9.139 | 0.079 | 0.000 |
| h2o | 32 | 9.215 | 0.083 | 0.062 |
| h2o | 64 | 9.341 | 0.071 | 0.146 |
| snap | 16 | 9.271 | 0.083 | 0.000 |
| snap | 32 | 9.358 | 0.075 | 0.000 |
| snap | 64 | 9.380 | 0.071 | 0.000 |

Distance 3000 filler bytes. Full cache 8.786 bits per byte, accuracy 0.108; control 8.833 bits, accuracy 0.104.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 8.602 | 0.088 | 0.000 |
| recency | 32 | 8.764 | 0.083 | 0.000 |
| recency | 64 | 8.758 | 0.088 | 0.000 |
| random | 16 | 8.532 | 0.092 | 0.006 |
| random | 32 | 8.585 | 0.092 | 0.020 |
| random | 64 | 8.623 | 0.088 | 0.037 |
| entropy | 16 | 8.819 | 0.083 | 0.000 |
| entropy | 32 | 8.746 | 0.092 | 0.017 |
| entropy | 64 | 8.561 | 0.100 | 0.068 |
| hybrid | 16 | 8.819 | 0.083 | 0.000 |
| hybrid | 32 | 8.885 | 0.083 | 0.000 |
| hybrid | 64 | 8.859 | 0.092 | 0.022 |
| h2o | 16 | 8.637 | 0.092 | 0.000 |
| h2o | 32 | 8.787 | 0.088 | 0.017 |
| h2o | 64 | 8.844 | 0.088 | 0.090 |
| snap | 16 | 8.807 | 0.088 | 0.000 |
| snap | 32 | 8.824 | 0.083 | 0.000 |
| snap | 64 | 8.869 | 0.088 | 0.000 |

## blt_tinystories_entropy_x10 (40 trials per distance)

Distance 400 filler bytes. Full cache 27.866 bits per byte, accuracy 0.079; control 28.031 bits, accuracy 0.083.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 26.702 | 0.083 | 0.000 |
| recency | 32 | 27.841 | 0.075 | 0.000 |
| recency | 64 | 27.995 | 0.058 | 0.000 |
| random | 16 | 27.882 | 0.067 | 0.062 |
| random | 32 | 27.866 | 0.071 | 0.127 |
| random | 64 | 27.638 | 0.075 | 0.310 |
| entropy | 16 | 27.357 | 0.079 | 0.042 |
| entropy | 32 | 27.474 | 0.075 | 0.173 |
| entropy | 64 | 27.360 | 0.083 | 0.451 |
| hybrid | 16 | 27.357 | 0.079 | 0.042 |
| hybrid | 32 | 26.881 | 0.092 | 0.119 |
| hybrid | 64 | 27.199 | 0.079 | 0.277 |
| h2o | 16 | 27.232 | 0.071 | 0.000 |
| h2o | 32 | 27.546 | 0.075 | 0.037 |
| h2o | 64 | 28.051 | 0.079 | 0.295 |
| snap | 16 | 27.202 | 0.088 | 0.000 |
| snap | 32 | 27.872 | 0.067 | 0.000 |
| snap | 64 | 28.477 | 0.075 | 0.090 |

Distance 1200 filler bytes. Full cache 28.695 bits per byte, accuracy 0.075; control 28.856 bits, accuracy 0.079.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 27.895 | 0.079 | 0.000 |
| recency | 32 | 28.204 | 0.075 | 0.000 |
| recency | 64 | 28.843 | 0.067 | 0.000 |
| random | 16 | 27.944 | 0.067 | 0.005 |
| random | 32 | 28.629 | 0.067 | 0.089 |
| random | 64 | 28.954 | 0.079 | 0.106 |
| entropy | 16 | 27.752 | 0.075 | 0.005 |
| entropy | 32 | 28.238 | 0.079 | 0.038 |
| entropy | 64 | 28.494 | 0.083 | 0.176 |
| hybrid | 16 | 27.752 | 0.075 | 0.005 |
| hybrid | 32 | 27.799 | 0.088 | 0.005 |
| hybrid | 64 | 28.372 | 0.083 | 0.074 |
| h2o | 16 | 27.633 | 0.071 | 0.000 |
| h2o | 32 | 28.263 | 0.092 | 0.000 |
| h2o | 64 | 28.174 | 0.079 | 0.277 |
| snap | 16 | 27.951 | 0.071 | 0.000 |
| snap | 32 | 28.808 | 0.067 | 0.000 |
| snap | 64 | 28.738 | 0.071 | 0.000 |

Distance 3000 filler bytes. Full cache 27.785 bits per byte, accuracy 0.083; control 27.761 bits, accuracy 0.083.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 28.673 | 0.063 | 0.000 |
| recency | 32 | 28.373 | 0.071 | 0.000 |
| recency | 64 | 28.358 | 0.067 | 0.000 |
| random | 16 | 27.972 | 0.079 | 0.006 |
| random | 32 | 28.363 | 0.083 | 0.020 |
| random | 64 | 28.503 | 0.067 | 0.037 |
| entropy | 16 | 28.035 | 0.088 | 0.000 |
| entropy | 32 | 27.833 | 0.067 | 0.017 |
| entropy | 64 | 28.534 | 0.075 | 0.068 |
| hybrid | 16 | 28.035 | 0.088 | 0.000 |
| hybrid | 32 | 28.871 | 0.075 | 0.000 |
| hybrid | 64 | 28.141 | 0.083 | 0.022 |
| h2o | 16 | 28.038 | 0.075 | 0.000 |
| h2o | 32 | 28.392 | 0.079 | 0.000 |
| h2o | 64 | 28.665 | 0.075 | 0.291 |
| snap | 16 | 28.488 | 0.075 | 0.000 |
| snap | 32 | 28.751 | 0.088 | 0.000 |
| snap | 64 | 28.388 | 0.071 | 0.000 |

## blt_wikitext103mix_entropy (40 trials per distance)

Distance 400 filler bytes. Full cache 7.467 bits per byte, accuracy 0.063; control 7.505 bits, accuracy 0.058.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 7.579 | 0.063 | 0.000 |
| recency | 32 | 7.540 | 0.058 | 0.000 |
| recency | 64 | 7.506 | 0.058 | 0.000 |
| random | 16 | 7.555 | 0.063 | 0.067 |
| random | 32 | 7.527 | 0.058 | 0.150 |
| random | 64 | 7.478 | 0.063 | 0.504 |
| entropy | 16 | 7.572 | 0.058 | 0.000 |
| entropy | 32 | 7.532 | 0.063 | 0.008 |
| entropy | 64 | 7.469 | 0.063 | 0.354 |
| hybrid | 16 | 7.572 | 0.058 | 0.000 |
| hybrid | 32 | 7.553 | 0.063 | 0.000 |
| hybrid | 64 | 7.491 | 0.063 | 0.117 |
| h2o | 16 | 7.559 | 0.058 | 0.000 |
| h2o | 32 | 7.540 | 0.063 | 0.000 |
| h2o | 64 | 7.492 | 0.063 | 0.367 |
| snap | 16 | 7.539 | 0.058 | 0.000 |
| snap | 32 | 7.483 | 0.054 | 0.079 |
| snap | 64 | 7.470 | 0.058 | 0.292 |

Distance 1200 filler bytes. Full cache 7.136 bits per byte, accuracy 0.079; control 7.135 bits, accuracy 0.079.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 7.201 | 0.079 | 0.000 |
| recency | 32 | 7.171 | 0.083 | 0.000 |
| recency | 64 | 7.141 | 0.079 | 0.000 |
| random | 16 | 7.182 | 0.079 | 0.037 |
| random | 32 | 7.191 | 0.083 | 0.062 |
| random | 64 | 7.170 | 0.067 | 0.144 |
| entropy | 16 | 7.247 | 0.079 | 0.000 |
| entropy | 32 | 7.251 | 0.071 | 0.000 |
| entropy | 64 | 7.210 | 0.079 | 0.040 |
| hybrid | 16 | 7.247 | 0.079 | 0.000 |
| hybrid | 32 | 7.213 | 0.075 | 0.000 |
| hybrid | 64 | 7.183 | 0.079 | 0.000 |
| h2o | 16 | 7.194 | 0.079 | 0.000 |
| h2o | 32 | 7.181 | 0.079 | 0.000 |
| h2o | 64 | 7.162 | 0.075 | 0.148 |
| snap | 16 | 7.231 | 0.079 | 0.000 |
| snap | 32 | 7.172 | 0.079 | 0.000 |
| snap | 64 | 7.122 | 0.088 | 0.000 |

Distance 3000 filler bytes. Full cache 7.337 bits per byte, accuracy 0.063; control 7.341 bits, accuracy 0.063.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 7.460 | 0.067 | 0.000 |
| recency | 32 | 7.432 | 0.067 | 0.000 |
| recency | 64 | 7.386 | 0.063 | 0.000 |
| random | 16 | 7.443 | 0.063 | 0.000 |
| random | 32 | 7.426 | 0.067 | 0.025 |
| random | 64 | 7.409 | 0.067 | 0.108 |
| entropy | 16 | 7.468 | 0.063 | 0.000 |
| entropy | 32 | 7.489 | 0.071 | 0.000 |
| entropy | 64 | 7.450 | 0.063 | 0.006 |
| hybrid | 16 | 7.468 | 0.063 | 0.000 |
| hybrid | 32 | 7.467 | 0.071 | 0.000 |
| hybrid | 64 | 7.432 | 0.063 | 0.000 |
| h2o | 16 | 7.448 | 0.063 | 0.000 |
| h2o | 32 | 7.429 | 0.067 | 0.000 |
| h2o | 64 | 7.381 | 0.067 | 0.317 |
| snap | 16 | 7.475 | 0.067 | 0.000 |
| snap | 32 | 7.419 | 0.067 | 0.000 |
| snap | 64 | 7.364 | 0.063 | 0.000 |

## blt_wikitext103mix_entropy_x10 (40 trials per distance)

Distance 400 filler bytes. Full cache 9.367 bits per byte, accuracy 0.054; control 9.397 bits, accuracy 0.054.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 9.269 | 0.058 | 0.000 |
| recency | 32 | 9.439 | 0.050 | 0.000 |
| recency | 64 | 9.463 | 0.046 | 0.000 |
| random | 16 | 9.159 | 0.046 | 0.067 |
| random | 32 | 9.302 | 0.042 | 0.150 |
| random | 64 | 9.267 | 0.038 | 0.504 |
| entropy | 16 | 8.925 | 0.050 | 0.000 |
| entropy | 32 | 8.833 | 0.042 | 0.008 |
| entropy | 64 | 8.891 | 0.042 | 0.354 |
| hybrid | 16 | 8.925 | 0.050 | 0.000 |
| hybrid | 32 | 9.176 | 0.063 | 0.000 |
| hybrid | 64 | 9.335 | 0.046 | 0.117 |
| h2o | 16 | 8.909 | 0.050 | 0.025 |
| h2o | 32 | 9.001 | 0.042 | 0.371 |
| h2o | 64 | 9.364 | 0.042 | 0.708 |
| snap | 16 | 9.509 | 0.042 | 0.000 |
| snap | 32 | 9.498 | 0.046 | 0.050 |
| snap | 64 | 9.464 | 0.050 | 0.588 |

Distance 1200 filler bytes. Full cache 8.489 bits per byte, accuracy 0.108; control 8.499 bits, accuracy 0.108.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 8.283 | 0.104 | 0.000 |
| recency | 32 | 8.420 | 0.096 | 0.000 |
| recency | 64 | 8.414 | 0.100 | 0.000 |
| random | 16 | 8.140 | 0.100 | 0.037 |
| random | 32 | 8.257 | 0.100 | 0.062 |
| random | 64 | 8.312 | 0.100 | 0.144 |
| entropy | 16 | 8.086 | 0.088 | 0.000 |
| entropy | 32 | 8.069 | 0.104 | 0.000 |
| entropy | 64 | 8.054 | 0.100 | 0.040 |
| hybrid | 16 | 8.086 | 0.088 | 0.000 |
| hybrid | 32 | 8.253 | 0.104 | 0.000 |
| hybrid | 64 | 8.369 | 0.104 | 0.000 |
| h2o | 16 | 8.124 | 0.100 | 0.050 |
| h2o | 32 | 8.139 | 0.108 | 0.471 |
| h2o | 64 | 8.216 | 0.108 | 0.729 |
| snap | 16 | 8.368 | 0.100 | 0.000 |
| snap | 32 | 8.432 | 0.100 | 0.000 |
| snap | 64 | 8.491 | 0.104 | 0.000 |

Distance 3000 filler bytes. Full cache 8.641 bits per byte, accuracy 0.092; control 8.644 bits, accuracy 0.096.

| Policy | Budget | Bits per byte | Accuracy | Name patches kept |
|---|---|---|---|---|
| recency | 16 | 8.453 | 0.108 | 0.000 |
| recency | 32 | 8.572 | 0.100 | 0.000 |
| recency | 64 | 8.553 | 0.104 | 0.000 |
| random | 16 | 8.264 | 0.108 | 0.000 |
| random | 32 | 8.280 | 0.100 | 0.025 |
| random | 64 | 8.313 | 0.104 | 0.108 |
| entropy | 16 | 8.184 | 0.100 | 0.000 |
| entropy | 32 | 8.198 | 0.108 | 0.000 |
| entropy | 64 | 8.167 | 0.113 | 0.006 |
| hybrid | 16 | 8.184 | 0.100 | 0.000 |
| hybrid | 32 | 8.389 | 0.108 | 0.000 |
| hybrid | 64 | 8.544 | 0.096 | 0.000 |
| h2o | 16 | 8.267 | 0.100 | 0.075 |
| h2o | 32 | 8.273 | 0.104 | 0.562 |
| h2o | 64 | 8.261 | 0.104 | 0.606 |
| snap | 16 | 8.509 | 0.100 | 0.000 |
| snap | 32 | 8.641 | 0.104 | 0.000 |
| snap | 64 | 8.666 | 0.092 | 0.000 |
